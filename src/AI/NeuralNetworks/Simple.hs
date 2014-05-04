{-|
  Simple parallel neural networks implementation

  @
  import AI.NeuralNetworks.Simple
  import Text.Printf
  import System.Random
  import Control.Monad
    
  calcXor net x y =
    let [r] = runNeuralNetwork net [x, y]
    in  r
    
  mse net =
    let square x = x * x
        e1 = square $ calcXor net 0 0
        e2 = square $ calcXor net 1 0 - 1
        e3 = square $ calcXor net 0 1 - 1
        e4 = square $ calcXor net 1 1
    in 0.5 * (e1 + e2 + e3 + e4)
    
  stopf best gnum = do
    let e = mse best
    when (gnum \`rem\` 100 == 0) $
      printf \"Generation: %02d, MSE: %.4f\\n\" gnum e
    return $ e \< 0.002 || gnum \>= 10000

  main = do
    gen \<- newStdGen
    let (randomNet, _) = randomNeuralNetwork gen [2,2,1] [Logistic, Logistic] 0.45
        examples = [ ([0,0],[0]), ([0,1],[1]), ([1,0],[1]), ([1,1],[0]) ]
    net \<- backpropagationBatchParallel randomNet examples 0.4 stopf :: IO (NeuralNetwork Double)
    putStrLn \"\"
    putStrLn $ \"Result: \" ++ show net
    _ \<- printf \"0 xor 0 = %.4f\\n\" (calcXor net 0 0)
    _ \<- printf \"1 xor 0 = %.4f\\n\" (calcXor net 1 0)
    _ \<- printf \"0 xor 1 = %.4f\\n\" (calcXor net 0 1)
    printf \"1 xor 1 = %.4f\" (calcXor net 1 1)
  @
-}
module AI.NeuralNetworks.Simple (
    ActivationFunction(..),
    NeuralNetwork,
    WeightDeltas,
    emptyNeuralNetwork,
    getWeights,
    setWeights,
    runNeuralNetwork,
    backpropagationOneStep,
    backpropagationStochastic,
    backpropagationBatchParallel,
    applyWeightDeltas,
    unionWeightDeltas,
    randomNeuralNetwork,
    crossoverCommon,
    crossoverMerge,
    mutationCommon
  ) where

import System.Random
import Data.List (unfoldr, foldl')
import qualified Data.Map.Strict as M
import qualified Data.IntMap.Strict as IM
import Control.Applicative
import Control.Arrow
import Control.DeepSeq
import Control.Parallel.Strategies
import GHC.Conc (numCapabilities)
import Data.List.Split (chunksOf)
import Data.Maybe (fromJust)
import Data.Word
import Data.Bits

-- | Activation function
data ActivationFunction = Tanh     -- ^ Hyperbolic tangent
                        | Logistic -- ^ Logistic function : 1 / (1 + exp (-x))
                        deriving (Show, Read, Eq)

logistic x = 1 / (1 + exp (-x))

applyAF Tanh = tanh
applyAF Logistic = logistic

applyAFDerivative Tanh x = let t = tanh x in (1 + t) * (1 - t) -- = 1 / (cosh x ^ (2 :: Int) )
applyAFDerivative Logistic x = let t = logistic x in t * (1 - t)

-- | Neural network
data NeuralNetwork a = NeuralNetwork [Word16] [ ActivationFunction ] (M.Map Word64 a)
                       deriving (Show, Read, Eq)

instance NFData a => NFData (NeuralNetwork a) where
    rnf (NeuralNetwork _ _ m) = rnf m `seq` ()

-- | Deltas calculated by backpropagation algorithm
newtype WeightDeltas a = WeightDeltas (M.Map Word64 a)
                         deriving (Show, Read, Eq)

instance NFData a => NFData (WeightDeltas a) where
    rnf (WeightDeltas m) = rnf m `seq` ()

-- | Neural network with all weights set to zero.
-- 
-- @
-- {- 
--    2 input neurons,
--    one hidden layer with 2 neurons and tanh activation function,
--    one output layer with 1 neuron and tanh activation function
-- -}
-- emptyNeuralNetwork [2, 2, 1] [Tanh, Tanh]
-- @
emptyNeuralNetwork :: [Word16]                -- ^ Number of neurons in each layer
                   -> [ ActivationFunction ]  -- ^ Activation functions
                   -> NeuralNetwork a         -- ^ New neural network
emptyNeuralNetwork sx ax =
    NeuralNetwork sx ax M.empty

-- | Weights of the given neural network.
getWeights :: NeuralNetwork a                 -- ^ Neural network
           -> [((Word16, Word16, Word16), a)] -- ^ Weights (layer 0.., neuron 1.., input 0..)
getWeights (NeuralNetwork _ _ wx) =
    map (first decodeKey) $ M.toList wx

-- | Change weights of the given neural network.
setWeights :: [((Word16, Word16, Word16), a)] -- ^ Weights
           -> NeuralNetwork a                 -- ^ Neural network
           -> NeuralNetwork a                 -- ^ Neural network with changed weights
setWeights lst (NeuralNetwork sx ax _) =
    let wx = M.fromList $ map (\((k1, k2, k3), v) -> (makeKey k1 k2 k3, v)) lst
    in  NeuralNetwork sx ax wx

-- | Run neural network.
runNeuralNetwork :: (Num a, Floating a) 
                 => NeuralNetwork a          -- ^ Neural network
                 -> [a]                      -- ^ Input signal
                 -> [a]                      -- ^ Output signal
runNeuralNetwork (NeuralNetwork sx ax m) input =
    let (result, _, _) = runNeuralNetwork' (head sx) (tail sx) ax 0 m [] [] input
    in result

runNeuralNetwork' _ [] _ _ _ ilfacc outacc xs = (xs, ilfacc, outacc)
runNeuralNetwork' _ _ [] _ _ _ _ _ = -- actually should never happen
    error "runNeuralNetwork' - invalid number of activation functions"
runNeuralNetwork' prevs (so:sx) (af:ax) layer ws ilfacc outacc xs =
    let ilfs = [ ( (layer, n), inducedLocalField n layer ws xs) | n <- [1..so] ]
        ilfacc' = ilfs ++ ilfacc
        outs = [ ( (layer, n), x ) | (x, n) <- zip xs [1..prevs] ]
        outacc' = outs ++ outacc
    in runNeuralNetwork' so sx ax (layer+1) ws ilfacc' outacc' (map (\(_, v) -> applyAF af v) ilfs)

inducedLocalField neuron layer ws xs =
    let weight k = getWeight (makeKey layer neuron k) ws
    in  weight 0 + sum [ weight i * x | (x, i) <- zip xs [1..] ]

-- | Run one step of the backpropagation algorithm.
backpropagationOneStep :: (Num a, Floating a)
                       => NeuralNetwork a   -- ^ Current neural network
                       -> a                 -- ^ Learning rate
                       -> [a]               -- ^ Input
                       -> [a]               -- ^ Expected output
                       -> WeightDeltas a    -- ^ Calculated deltas
backpropagationOneStep (NeuralNetwork sx ax wx) learningRate input expout =
    let (result, inducedLocalFields, outputs) = runNeuralNetwork' (head sx) (tail sx) ax 0 wx [] [] input
        errors = [ d - o | (d, o) <- zip expout result ]
        inducedLocalFieldsMap = M.fromList inducedLocalFields
        outputsMap = M.fromList outputs
        deltasMap = calculateDeltas sx ax wx errors inducedLocalFieldsMap
        wdx = M.mapWithKey 
                (\k _ ->
                    let (ln, n, i) = decodeKey k
                        out = if i == 0 then 1
                                        else fromJust $ M.lookup (ln, i) outputsMap
                    in learningRate * out * fromJust (M.lookup (ln, n) deltasMap)
                ) wx
    in WeightDeltas wdx

-- | Run backpropagation algorithm in stochastic mode.
backpropagationStochastic :: (Num a, Floating a)
                          => NeuralNetwork a                        -- ^ Neural network
                          -> [([a],[a])]                            -- ^ Trainset: inputs and expected outputs
                          -> a                                      -- ^ Learning rate
                          -> (NeuralNetwork a -> Int -> IO Bool)    -- ^ Stop function, 1st arg - current NN, 2nd arg - generation number
                          -> IO (NeuralNetwork a)                   -- ^ Trained neural network
backpropagationStochastic net0 set0 learningRate stopf = do
    g0 <- newStdGen
    run g0 net0 set0 0
    where
        len = length set0
        run rg net set gnum = do
            let (rg', set') = shuffleList rg len set
                net' = foldl' (\n (i, o) -> let wdx = backpropagationOneStep n learningRate i o
                                            in applyWeightDeltas wdx n) net set'
            stop <- stopf net' gnum
            if stop then return net'
                    else run rg' net' set' (gnum+1)

-- | Run backpropagation algorithm in batch mode. This code runs faster in parallel, so don't forget to use +RTS -N.
backpropagationBatchParallel :: (Num a, Floating a, NFData a) 
                             => NeuralNetwork a                     -- ^ Neural network
                             -> [([a],[a])]                         -- ^ Trainset: inputs and expected outputs
                             -> a                                   -- ^ Learning rate
                             -> (NeuralNetwork a -> Int -> IO Bool) -- ^ Stop function, 1st arg - current NN, 2nd arg - generation number
                             -> IO (NeuralNetwork a)                -- ^ Trained neural network
backpropagationBatchParallel net0 set learningRate stopf =
    run net0 0
    where
        chunks = chunksOf ( ceiling $ fromIntegral (length set) / (fromIntegral numCapabilities :: Double) ) set
        run net gnum = do
            let wdx = map (unionWeightDeltas . map (uncurry $ backpropagationOneStep net learningRate)) chunks
                        `using` parList rdeepseq
                totalWdx = unionWeightDeltas wdx
                net' = applyWeightDeltas totalWdx net
            stop <- stopf net' gnum
            if stop then return net'
                    else run net' (gnum+1)

-- | Apply deltas to the neural netwotk.
applyWeightDeltas :: (Num a, Floating a)
                  => WeightDeltas a     -- ^ Deltas
                  -> NeuralNetwork a    -- ^ Neural network
                  -> NeuralNetwork a    -- ^ Neural network with updated weights
applyWeightDeltas (WeightDeltas dwx) (NeuralNetwork sx ax wx) =
    let wx' = M.mapWithKey (\k w -> w + fromJust (M.lookup k dwx)) wx
    in NeuralNetwork sx ax wx'

-- | Union list of deltas into one WeightDeltas.
unionWeightDeltas :: (Num a, Floating a)
                  => [WeightDeltas a]   -- ^ List of WeightDeltas
                  -> WeightDeltas a     -- ^ United WeightDeltas
unionWeightDeltas [] = error "Empty list"
unionWeightDeltas [x] = x
unionWeightDeltas (WeightDeltas hd : tl) =
    let tm = foldl' (\acc (WeightDeltas m) -> M.mapWithKey (\k w -> w + fromJust (M.lookup k m)) acc) hd tl
    in WeightDeltas tm

calculateDeltas sx ax wx errors ilfm =
    let (s:sx') = reverse sx
        (a:ax') = reverse ax
        cl = fromIntegral $ length sx - 2
        acc = M.fromList [ ((cl, n), err * applyAFDerivative a (fromJust $ M.lookup (cl, n) ilfm )) | (err, n) <- zip errors [1..s] ]
    in calculateDeltas' (cl - 1) s sx' ax' wx ilfm acc

calculateDeltas' _ _ _ [] _ _ acc = acc
calculateDeltas' cl sprev sx ax wx ilfm acc = 
    let (s:sx') = sx
        (a:ax') = ax
        err n = sum [ fromJust $ (*) <$> M.lookup (cl+1, k) acc <*> M.lookup (makeKey (cl+1) k n) wx | k <- [1..sprev] ]
        newDeltas = [ ((cl, n), err n * applyAFDerivative a (fromJust $ M.lookup (cl, n) ilfm)) | n <- [1..s] ] 
        acc' = foldl' (\m (k, v) -> M.insert k v m) acc newDeltas
    in calculateDeltas' (cl - 1) s sx' ax' wx ilfm acc'

-- | Generate random neural network.
randomNeuralNetwork :: (RandomGen g, Random a, Num a, Ord a)
                    => g                        -- ^ RandomGen
                    -> [Word16]                 -- ^ Number of neurons in each layer
                    -> [ ActivationFunction ]   -- ^ Activation functions
                    -> a                        -- ^ Maximum weight; all weights in NN will be between -maxw and maxw
                    -> (NeuralNetwork a, g)     -- ^ Random neural network and new RandomGen
randomNeuralNetwork gen sx ax maxw 
    | length sx /= length ax + 1 = error "Number of layers and activation functions mismatch"
    | maxw < 0 = randomNeuralNetwork gen sx ax (-maxw)
    | otherwise =
        let keys = generateKeys sx
            (weights, gen') = generateWeights gen maxw
            ws = M.fromList $ zip keys weights
        in  (NeuralNetwork sx ax ws, gen')

makeKey :: Word16 -> Word16 -> Word16 -> Word64
makeKey layer n i =
    let t1 = fromIntegral layer
        t2 = fromIntegral n
        t3 = fromIntegral i
    in shiftL t1 32 .|. shiftL t2 16 .|. t3

decodeKey :: Word64 -> (Word16, Word16, Word16)
decodeKey k =
    let t1 = fromIntegral $ shiftR k 32
        t2 = fromIntegral $ shiftR k 16 .&. 0xFFFF
        t3 = fromIntegral $ k .&. 0xFFFF
    in (t1, t2, t3)

generateKeys sx =
    [ makeKey layer n i | (layer, inputs, neurons) <- zip3 [0..] (init sx) (tail sx), n <- [1 .. neurons], i <- [0 .. inputs ] ]

generateWeights gen maxw =
    let (gen1, gen2) = split gen
    in (unfoldr (Just . randomR (-maxw, maxw) ) gen1, gen2)

-- | Crossover of two neural networks.
crossoverCommon :: (Num a, RandomGen g)
                => g                        -- ^ RandomGen
                -> NeuralNetwork a          -- ^ First neural network
                -> NeuralNetwork a          -- ^ Second neural network
                -> ([NeuralNetwork a],g)    -- ^ Children and new RandomGen
crossoverCommon g0 (NeuralNetwork sx1 ax1 wx1) (NeuralNetwork _ _ wx2) =
    let keys = generateKeys sx1
        (idx, g1) = randomR (1, length keys - 1) g0
        (keys1, keys2) = splitAt idx keys
        tmpMap wx lst = M.fromList [ (k, getWeight k wx) | k <- lst ]
        wx1' = tmpMap wx1 keys1 `M.union` tmpMap wx2 keys2
        wx2' = tmpMap wx1 keys2 `M.union` tmpMap wx2 keys1
    in ( [ NeuralNetwork sx1 ax1 wx1', NeuralNetwork sx1 ax1 wx2' ], g1)

-- | Another implementation of crossover. Weights of a child are just some function of corresponding parent weights.
crossoverMerge :: (Num a, RandomGen g)
               => (a -> a -> a)         -- ^ Mentioned 'some function'
               -> g                     -- ^ Not used
               -> NeuralNetwork a       -- ^ First neural network
               -> NeuralNetwork a       -- ^ Second neural netwrok
               -> ([NeuralNetwork a],g) -- ^ Children (actually - exactly one child) and exact copy of the 2nd argument
crossoverMerge avgf gen (NeuralNetwork sx1 ax1 wx1) (NeuralNetwork _ _ wx2) =
    let wx' = M.fromList [ (k, getWeight k wx1 `avgf` getWeight k wx2) | k <- generateKeys sx1]
    in  ( [ NeuralNetwork sx1 ax1 wx' ], gen )

-- | Mutate given neural netwrok.
mutationCommon :: (Random a, Num a, RandomGen g)
               => Double                -- ^ Percent of mutating weights, (0.0; 1.0)
               -> a                     -- ^ Maximum weight, mutated weights will be between -maxw and maxw
               -> g                     -- ^ RandomGen
               -> NeuralNetwork a       -- ^ Neural network
               -> (NeuralNetwork a, g)  -- ^ New neural network and RandomGen
mutationCommon percent maxw gen (NeuralNetwork sx ax wx) =
    let layers = length sx - 1
        mutnum = truncate $ percent * fromIntegral (M.size wx) :: Int
        (wx', gen') = mutationCommon' mutnum (abs maxw) gen wx (init sx) (tail sx) layers
    in (NeuralNetwork sx ax wx', gen')

mutationCommon' mutnum maxw g0 wx inputs outputs layers
    | mutnum <= 0 = (wx, g0)
    | otherwise =
        let (layer, g1) = randomR (0, layers - 1) g0
            (neuron, g2) = randomR (1, outputs !! layer) g1
            (weightIdx, g3) = randomR (0, inputs !! layer) g2
            (newWeight, g4) = randomR (- maxw, maxw) g3
            wx' = M.insert (makeKey (fromIntegral layer) neuron weightIdx) newWeight wx
        in mutationCommon' (mutnum - 1) maxw g4 wx' inputs outputs layers

getWeight :: (Num a, Ord k) => k -> M.Map k a -> a
getWeight = M.findWithDefault 0

shuffleList :: (RandomGen g) => g -> Int -> [a] -> (g, [a])
shuffleList g lstlen lst =
    shuffleList' g (lstlen-1) (lstlen-1) (IM.fromList $ zip [0..] lst)

shuffleList' g maxpos step m0
    | step < 0 = (g, map snd $ IM.toList m0)
    | otherwise =
        let (pos, g') = randomR (0, maxpos) g
            v1 = fromJust $ IM.lookup step m0
            v2 = fromJust $ IM.lookup pos m0
            m1 = IM.insert step v2 m0
            m2 = IM.insert pos v1 m1
        in shuffleList' g' maxpos (step-1) m2


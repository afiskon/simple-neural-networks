import AI.NeuralNetworks.Simple
import Text.Printf
import System.Random
import Control.Monad

calcXor net x y =
    let [r] = runNeuralNetwork net [x, y]
    in  r

mse net =
    let square x = x * x
        e1 = square $ calcXor net (-1) (-1) - (-1)
        e2 = square $ calcXor net   1  (-1) -   1
        e3 = square $ calcXor net (-1)   1  -   1
        e4 = square $ calcXor net   1    1  - (-1)
    in 0.5 * (e1 + e2 + e3 + e4)

stopf best gnum = do
    let e = mse best
    when (gnum `rem` 100 == 0) $
        printf "Generation: %02d, MSE: %.4f\n" gnum e
    return $ e < 0.002 || gnum >= 10000

main = do
    gen <- newStdGen
    let (randomNet, _) = randomNeuralNetwork gen [2,2,1] [Tanh, Tanh] 0.25
        examples = [ ([-1,-1],[-1]), ([-1,1],[1]), ([1,-1],[1]), ([1,1],[-1]) ]
    net <- backpropagationBatchParallel randomNet examples 0.4 stopf :: IO (NeuralNetwork Double)
    putStrLn ""
    putStrLn $ "Result: " ++ show net
    _ <- printf "0 xor 0 = %.4f\n" (calcXor net (-1) (-1))
    _ <- printf "1 xor 0 = %.4f\n" (calcXor net 1 (-1))
    _ <- printf "0 xor 1 = %.4f\n" (calcXor net (-1) 1)
    printf "1 xor 1 = %.4f" (calcXor net 1 1)

let xs = [2.69; 2.53; 1.05; 0.83; 0.49; 0.31]

let powOp x = (10.0 ** 5.0 * x) ** 2.0
let sCalc f xs ys = (ys |> List.sumBy f) / (xs |> List.sumBy f)


sCalc powOp xs [0.49; 0.39]

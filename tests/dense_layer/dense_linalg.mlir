#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @model(%arg0: tensor<1x10xf32>, %arg1: tensor<10x5xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5x1xf32>, %arg4: tensor<1xf32>) -> tensor<1x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x5xf32>) -> tensor<1x5xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1x10xf32>, tensor<10x5xf32>) outs(%1 : tensor<1x5xf32>) -> tensor<1x5xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<1x5xf32>, tensor<5xf32>) outs(%1 : tensor<1x5xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %12 = arith.addf %in, %in_3 : f32
      linalg.yield %12 : f32
    } -> tensor<1x5xf32>
    %4 = tensor.empty() : tensor<1x5xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1x5xf32>) outs(%4 : tensor<1x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.maximumf %in, %cst_0 : f32
      linalg.yield %12 : f32
    } -> tensor<1x5xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %6 = tensor.empty() : tensor<1x1xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %8 = linalg.matmul ins(%5, %arg3 : tensor<1x5xf32>, tensor<5x1xf32>) outs(%7 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %arg4 : tensor<1x1xf32>, tensor<1xf32>) outs(%7 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %12 = arith.addf %in, %in_3 : f32
      linalg.yield %12 : f32
    } -> tensor<1x1xf32>
    %10 = tensor.empty() : tensor<1x1xf32>
    %cst_2 = arith.constant 1.000000e+00 : f32
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<1x1xf32>) outs(%10 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.negf %in : f32
      %13 = math.exp %12 : f32
      %14 = arith.addf %cst_2, %13 : f32
      %15 = arith.divf %cst_2, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<1x1xf32>
    return %11 : tensor<1x1xf32>
  }
}

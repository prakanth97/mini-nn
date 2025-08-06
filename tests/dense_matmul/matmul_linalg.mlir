#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @model(%arg0: tensor<1x10xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<5xf32>) -> tensor<1x5xf32> {
    %0 = tensor.empty() : tensor<10x5xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<5x10xf32>) outs(%0 : tensor<10x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<10x5xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<1x5xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x5xf32>) -> tensor<1x5xf32>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<1x10xf32>, tensor<10x5xf32>) outs(%3 : tensor<1x5xf32>) -> tensor<1x5xf32>
    %5 = tensor.empty() : tensor<1x5xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg2 : tensor<1x5xf32>, tensor<5xf32>) outs(%5 : tensor<1x5xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<1x5xf32>
    return %6 : tensor<1x5xf32>
  }
}

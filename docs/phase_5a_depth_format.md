# COLMAP Dense Format Notes

Reconevery's implementation follows the official COLMAP dense reconstruction
format and script layout.

## Arrays

The ASCII prefix contains three positive decimal integers separated and
terminated by `&`. The remaining bytes are exactly
`width * height * channels * 4` little-endian float32 bytes. COLMAP writes the
array in Fortran order. Reconevery exposes logical `(row, column, channel)`
coordinates and rejects trailing or missing data.

Depth values must be finite and positive before backprojection. A zero, NaN, or
infinite value is not measured geometry. Normal maps must have three finite
channels; retained normals are rotated into the COLMAP world and normalized.

## Consistency graphs

After the header, each little-endian int32 record is:

```text
column, row, N, source_index_0, ..., source_index_(N-1)
```

This order follows official COLMAP `ConsistencyGraph::InitializeMap`; it is not
the more common row-first array notation. Rows and columns must lie inside the dense map. Source indices address the
dense workspace image list, not canonical frame IDs. The typed workspace
manifest provides the reversible map.

## Mask mapping

Masks originate in normalized distorted-frame coordinates. The output dense
pixel is mapped into the source camera with the original distortion parameters.
Nearest-neighbor resampling preserves binary semantics. A mask is resized to a
depth-map resolution only when the official depth map is lower resolution than
the undistorted RGB, again with nearest-neighbor sampling.

The RGB remap check catches crop, scale, principal-point, and distortion-policy
drift before any object point is accepted.

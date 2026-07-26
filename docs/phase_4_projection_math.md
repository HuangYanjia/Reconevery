# Phase 4 Projection Math

## Coordinate Contract

Reconevery camera poses are:

```text
transform_world_from_camera
quaternion order xyzw
camera axes x-right, y-down, z-forward
world frame colmap_arbitrary
linear units arbitrary_units
```

For a camera pose `(R_wc, t_wc)`:

```text
p_world = R_wc p_camera + t_wc
R_cw = transpose(R_wc)
t_cw = -R_cw t_wc
p_camera = R_cw p_world + t_cw
```

Quaternion input is normalized before conversion. No gravity or metric transform is introduced.

## Pinhole Projection

For `p_camera=(x,y,z)` and `z>0`:

```text
u = fx * x / z + cx
v = fy * y / z + cy
```

Points with `z<=0` are behind the camera and are not projected. Principal point offsets and
non-square `fx/fy`, width, and height are retained.

Pixel centers map to nvdiffrast normalized device coordinates as:

```text
x_ndc = 2 * (u + 0.5) / width - 1
y_ndc = 1 - 2 * (v + 0.5) / height
```

The Y sign change is explicit because OpenCV image Y points down while raster NDC Y points up.
NDC conversion has a tested inverse.

For positive camera Z and scene-relative near/far:

```text
A = (far + near) / (far - near)
B = -2 * far * near / (far - near)
clip = [x_ndc*z, y_ndc*z, A*z+B, z]
```

Near and far use the global scene diagonal and positive camera-depth quantiles. They are arbitrary
COLMAP units, never meters.

## Distortion

Normalized coordinates use:

```text
r2 = x*x + y*y
radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
x_d = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
y_d = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
```

Mappings are:

- `SIMPLE_PINHOLE`, `PINHOLE`: no distortion;
- `SIMPLE_RADIAL`: `k1`;
- `RADIAL`: `k1,k2`;
- `OPENCV`: `k1,k2,p1,p2` and optional `k3`.

Real lifting uses OpenCV's deterministic rectification map and nearest-neighbor mask remap. The
map bytes and output dimensions are hashed. Direct distortion functions and projection/rotation
round trips are covered independently by unit tests.

## Visibility

nvdiffrast returns a local candidate-triangle index at each nearest visible pixel. Reconevery
maps it through the frustum-candidate array to the original GenRecon global face ID. It never
uses a decimated proxy for canonical attribution.

Synthetic reference tests use a pure-Python barycentric Z-buffer. Two overlapping triangles
prove that the front triangle wins and the hidden back face receives no evidence. Whole-scene
rotation tests preserve face identity while retaining the `colmap_arbitrary` label.

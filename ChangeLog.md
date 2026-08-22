# Fix `aglobal`

## Issue

User:

```
Concerning the gravity computation, it is, in the original Madgwick thesis corresponding to the third row of the transformation matrix, which maps the gravity vector into the orientation of the sensor. There's an easy to miss detail in there. The third row doesn't map to the gravity vector, it maps to the unit vector [0, 0, 1], which is the negative of the gravity vector (because of the choice to do all the math in g's instead of meters per second). The feedback system works as one would want it to, but this has the consequence of the internal quaternion (assuming it hasn't reoriented due to sensor movement) mapping the standard basis [1, 0, 0], [0, 1, 0], [0, 0, 1] to [1, 0, 0], [0, -1, 0], [0, 0, -1] instead of being a unity transform.

Perhaps less importantly, when the sensor is at rest, the gravity vector estimate starts at [0, 0, 1] and slowly flips while the system settles. I don't think this really matters as it makes assumptions about the initial state that aren't always true, but this flipping is the process by which the feedback system makes the quaternion upside down.

The problem is easily fixable by popping a negative in front of the gravity vector, your half_gravity array, I suppose.

The gravity_sensor() result and half_gravity() result are the same down to a factor of two, as the name indicates, but it takes a little bit fiddling with quaternion identities to see that that is true. In any case, I don't think there's much need to have both.

The problem expresses itself further if you check the aglobal vector, which is incorrectly transformed.
```

## Required fixes

1. Add a failing `aglobal` regression test for `Madgwick`.

   - Use a non-identity orientation containing roll, pitch, and yaw.
   - Choose a known Earth-frame linear-acceleration vector.
   - Rotate gravity plus linear acceleration into the sensor frame.
   - Confirm `azero` equals the expected sensor-frame linear acceleration.
   - Confirm `aglobal` equals the original Earth-frame vector.
   - Avoid a zero/stationary-only test because either rotation direction passes it.

2. Add the equivalent regression test for `Fusion`.

   - Exercise the pure-Python path.
   - Exercise the compiled `_fcore` path when available.
   - Require both paths to produce equivalent results.

3. Fix the Madgwick sensor-to-Earth transform in `pyIMU/madgwick.py`:

   ```python
   aq = self.q * Quaternion(self.azero) * self.q.conjugate
   ```

4. Fix the pure-Python Fusion transform in `pyIMU/fusion.py` using the same quaternion order.

5. Fix the optimized Fusion transform in `pyIMU/_fcore.pyx`.

   - Implement `q * azero * q.conjugate`.
   - Keep it mathematically identical to the Python implementation.

6. Regenerate the checked-in `pyIMU/_fcore.c`.

   - This ensures installations built without Cython receive the correction.
   - Verify the generated diff is limited to the expected Cython output.

7. Clarify the output documentation.

   - State that `azero` is gravity-compensated acceleration in the sensor/body frame.
   - State that `aglobal` is that vector rotated from sensor/body coordinates into NED Earth coordinates.
   - State that acceleration outputs remain expressed in `g`.

## Valid clarification, but no behavioral change

8. Document the accelerometer and gravity convention more explicitly.

   - Earth coordinates are NED.
   - The reference down vector is `[0, 0, +1]`.
   - An aligned stationary input is expected to be `[0, 0, +1] g`.
   - Sensors producing another polarity or axis arrangement must be remapped before being passed to the filter.

9. Explain the relationship between `half_gravity` and `gravity_sensor`.

   - For a normalized quaternion, `gravity_sensor == 2 * half_gravity`.
   - Retain `half_gravity` because the Fusion feedback equations directly use the half-scaled form.
   - Retain `gravity_sensor` for clear gravity subtraction and acceleration-output calculation.

## Explicitly out of scope

Do not:

- Negate `half_gravity`.
- Negate `gravity_sensor`.
- Change the Madgwick objective-function signs.
- Change the NED convention.
- Remove either gravity helper solely because they are related by a factor of two.
- Change sensor-axis handling without evidence of an actual input-convention mismatch.

## Verification

10. Run the complete suite with the selected environment:

    ```bash
    source ~/Python/pyqt/bin/activate
    python -m pip install -e .
    python -m pytest
    ```

11. Rebuild the accelerated modules and rerun the suite:

    ```bash
    python setup.py build_ext --inplace
    python -m pytest
    ```

12. Numerically compare Python and accelerated Fusion outputs for the same tilted-orientation case.

## Completion criteria

The work is complete when:

- The new tests fail with the current inverse transform.
- The new tests pass after the transform correction.
- The pure-Python and accelerated Fusion results agree.
- All existing tests continue to pass.

## Completion status

The `aglobal` implementation and verification work was completed on 2026-08-22:

- Corrected the sensor/body-to-NED Earth transform in Madgwick, Python Fusion, and compiled Fusion.
- Added tilted-orientation regression coverage for Madgwick and both Fusion backends.
- Regenerated and rebuilt `_fcore.c` from the corrected `_fcore.pyx` implementation.
- Confirmed all 18 tests pass with the rebuilt accelerated modules.
- Confirmed Python and `_fcore` results agree within `1.804e-16` for the tilted test case.

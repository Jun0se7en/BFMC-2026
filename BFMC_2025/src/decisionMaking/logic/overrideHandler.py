from src.decisionMaking.control.control_state import CONTROL_STATE
from src.decisionMaking.control.vehicle_state import STATE, ObjectFlag
import time

class OverrideHandler:
    def __init__(self):
        self.sequence = []
        self.current_step = 0
        self.step_end = time.monotonic()
        self.paused = False

    def _set_sequence(self, steps):
        self.sequence = steps
        self.current_step = 0
        self.paused = False
        self._apply_current_step()
        
    def is_active(self) -> bool:
        return self.current_step < len(self.sequence)
    
    def _apply_current_step(self):
        if self.current_step < len(self.sequence):
            speed, angle, duration = self.sequence[self.current_step]
            CONTROL_STATE.set_override(speed=int(speed), angle=int(angle), duration=duration)
            self.step_end = time.monotonic() + duration
            print(f"[STEP {self.current_step}] speed={speed}, angle={angle}, duration={duration}")

    def update(self, emergency_active=False):
        # Use for emergency handling that needs braking
        if emergency_active:
            if not self.paused:
                CONTROL_STATE.cancel_override()
                CONTROL_STATE.update(speed=0.0, angle=0.0)
                self.paused = True
                print("🚨 Emergency pause")
            return True  # Still running, but paused

        if self.paused:
            self.paused = False
            remaining = self.step_end - time.monotonic()
            if remaining > 0:
                speed, angle, _ = self.sequence[self.current_step]
                CONTROL_STATE.set_override(speed=speed, angle=angle, duration=remaining)
                print("✅ Resuming override")

        if time.monotonic() >= self.step_end:
            self.current_step += 1
            if self.current_step < len(self.sequence):
                self._apply_current_step()
            else:
                # print("🏁 Sequence complete")
                return False  # Finished

        # Actively write to CONTROL_STATE even during ongoing step
        if not self.paused and self.current_step < len(self.sequence):
            speed, angle, _ = self.sequence[self.current_step]
            CONTROL_STATE.update(speed=speed, angle=angle)

        return True  # Still running

    def override_stop(self):
        self._set_sequence([
            (0, 0.0, 3.0),  # Stop for 2 seconds
            (25, 0, 1)
        ])  

    def override_forward(self):
        self._set_sequence([
            (0, 0, 2),
            (25, 0, 5.6),
            (0, 0, 1)
        ])

    def override_right(self):
        self._set_sequence([
            (0, 0, 2),
            (0, 7, 0.3),
            (0, 14, 0.3),
            (0, 17, 0.3),
            (25, 24, 3),
            (0, 0, 1),
        ])

    def override_left(self):
        self._set_sequence([
            (0, 0.0, 2),
            (0, -7, 0.1),
            (25, 0, 3),
            (25, -24, 2.2),
            (25, -14, 1),
            (0, 0, 1)
        ])

    def override_roundabout_right(self):
        self._set_sequence([
            (0, 0, 2),
            (0, 5, 0.1),
            (0, 15, 0.1),
            (25, 24, 2),
            (25, 0, 2),
            (0, 5, 0.1),
            (0, 15, 0.1),
            (25, 24, 2),
            (0, 0, 1),
        ])


    # def override_roundabout_forward(self):
    #     self._set_sequence([
    #         (0, 0.0, 2),
    #         (25, 0, 2),
    #         # (25, 3, 0.3),
    #         # (25, 7, 0.3),
    #         # (25, 10, 0.3),
    #         # (25, 16, 0.3),
    #         (25, 24, 2),
    #         # (25, 0, 1),
    #         (25, -24, 2.5),
    #         (25, 24, 1),
    #         # (25, 3, 0.3),
    #         # (25, 7, 0.3),
    #         # (25, 10, 0.3),
    #         # (25, 16, 0.3),
    #         (0, 0, 1)
    #     ])

    def parking(self):
        self._set_sequence([
            (0, 0, 1),
            (25, 0, 4),
            (25, 3, 0.2),
            (25, 7, 0.3),
            (25, 15, 0.3),
            (0, 0, 1),
            (-20, 20, 3.0),
            (-20, -24, 2),
            (25, 5, 1),
            (20, 0, 3)
        ])

    def out_parking(self):
        self._set_sequence([
            (-20, -10, 1.0),
            (20, 24, 1.5),
            (20, -24, 2.0),
            (-20, 0.0, 2.0),
            (0, 0.0, 1.0)
        ])

    def left_overtake(self):
        self._set_sequence([
            (35, -24, 2.0),
            (35, 24, 0.5),
            (35, 0.0, 2.5),
            (35, 24, 1.0)
        ])

    def right_overtake(self):
        self._set_sequence([
            (35, 24, 2.0),
            (35, -24, 0.5),
            (35, 0.0, 2.5),
            (35, -24, 1.0)
        ])
        
    def handle_intersection(self, enable_intersection: bool, direction: str):
        if enable_intersection:
            # direction = direction.lofawer()
            if direction == "left":
                self.override_left()
            elif direction == "right":
                self.override_right()
            elif direction == "forward":
                self.override_forward()
        
if __name__ == "__main__":
    handler = OverrideHandler()
    handler.parking()

    t0 = time.monotonic()

    while handler.update(emergency_active=(3 <= time.monotonic() - t0 < 5)):
        now = time.monotonic() - t0

        # Simulate pedestrian detection between t = 3s and t = 5s
        if 3 <= now < 5:
            STATE.set("sign_state", ObjectFlag.PEDESTRIAN)
            handler.update(emergency_active=True)
        else:
            STATE.clear("sign_state", ObjectFlag.PEDESTRIAN)
            handler.update(emergency_active=False)

        # Print current speed and angle
        speed, angle, _ = CONTROL_STATE.snapshot()
        print(f"[t={now:.1f}s] Speed: {speed}, Angle: {angle}")

        time.sleep(0.1)

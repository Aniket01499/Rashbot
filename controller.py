from pynput.keyboard import Key, Controller as KeyboardController

class SteeringController:
    def __init__(self):
        self.keyboard = KeyboardController()
        
        # Tuned up slightly for solid, committed turns
        self.Kp = 1.0 
        
        # High derivative acts as an "Early Release Brakes" to prevent wobble
        self.Kd = 12 
        
        self.prev_error = 0
        self.current_key = None
        self.ego_lost_frames = 0 

        # --- NEW: Immediately flush the OS buffer on startup ---
        self.flush_keys()

    def flush_keys(self):
        # Forcefully tell the OS to lift all fingers off the keys
        self.keyboard.release(Key.left)
        self.keyboard.release(Key.right)
        self.keyboard.release(Key.up)
        self.current_key = None

    def calculate_steering(self, target_x, ego_x):
        # 1. GHOST FRAME BUFFER
        if target_x is None or ego_x is None:
            self.ego_lost_frames += 1
            if self.ego_lost_frames > 5: 
                self.emergency_stop()
            return
            
        self.ego_lost_frames = 0 
        
        # 2. THE HARDWARE HEARTBEAT (Throttle)
        self.keyboard.press(Key.up)
        
        # 3. PREDICTIVE STEERING LOGIC
        error = target_x - ego_x
        derivative = error - self.prev_error
        
        # Force = (Distance to line) + (Speed of approach)
        steering_force = (self.Kp * error) + (self.Kd * derivative)
        
        self.prev_error = error
        self.actuate_keys(steering_force)

    def actuate_keys(self, force):
        # ==========================================
        # PREDICTIVE HOLD LOGIC
        # ==========================================
        # If the force is below 15, we are either close to the line, 
        # or we are approaching it fast enough that we should coast.
        DEADZONE = 15 

        # Coasting (Let the game auto-center the bike)
        if abs(force) < DEADZONE: 
            if self.current_key is not None:
                self.keyboard.release(self.current_key)
                self.current_key = None
            return

        # Target is to the LEFT
        if force < -DEADZONE: 
            if self.current_key != Key.left:
                if self.current_key == Key.right:
                    self.keyboard.release(Key.right)
                self.keyboard.press(Key.left)
                self.current_key = Key.left
            else:
                # Heartbeat refresh for Wine emulators
                self.keyboard.press(Key.left) 
                
        # Target is to the RIGHT
        elif force > DEADZONE: 
            if self.current_key != Key.right:
                if self.current_key == Key.left:
                    self.keyboard.release(Key.left)
                self.keyboard.press(Key.right)
                self.current_key = Key.right
            else:
                # Heartbeat refresh for Wine emulators
                self.keyboard.press(Key.right)

    def emergency_stop(self):
        if self.current_key is not None:
            self.keyboard.release(self.current_key)
            self.current_key = None
            
        self.keyboard.release(Key.up)
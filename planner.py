import numpy as np
import cv2

class APFPlanner:
    def __init__(self):
        self.SCREEN_W = 640
        self.SCREEN_H = 480
        self.SCREEN_CENTER = 320
        
        self.HORIZON_TOP = 150
        self.HORIZON_BOTTOM = 350
        
        self.BOT_Y = 400 

        # THE EGO ZONE (15%-85% X, 20%-70% Y)
        self.EGO_X_MIN = self.SCREEN_W * 0.15 
        self.EGO_X_MAX = self.SCREEN_W * 0.85 
        self.EGO_Y_MIN = self.SCREEN_H * 0.20 
        self.EGO_Y_MAX = self.SCREEN_H * 0.70 

        self.BIKE_ID = 0
        self.CAR_ID = 1
        self.ROAD_ID = 3

        # RACING TUNING
        # RETUNED: K_REP lowered to 50 because we are now multiplying it by the car's bounding box width!
        self.K_REP = 50    
        self.K_EDGE = 80000 
        
        self.alpha = 0.1      
        self.smoothed_target_x = 320 
        self.smoothed_left = 100
        self.smoothed_right = 540
        
        self.last_ego_x = 320 
        self.last_ego_y = 400 
        self.last_safe_x = 320

    def extract_scene_data(self, boxes):
        obstacles = []
        road_boxes = []
        possible_egos = []
        ego_box = None
        
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            cls_id = int(cls)
            
            if cls_id == self.ROAD_ID:
                road_boxes.append(box)
            elif cls_id == self.CAR_ID:
                # FIX 1: We removed the Horizon limits. 
                # The AI now considers EVERY car on the screen as a threat.
                obstacles.append(box)
            elif cls_id == self.BIKE_ID:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if (self.EGO_X_MIN <= cx <= self.EGO_X_MAX) and (self.EGO_Y_MIN <= cy <= self.EGO_Y_MAX):
                    possible_egos.append(box)
                else:
                    obstacles.append(box)

        if possible_egos:
            if len(possible_egos) == 1:
                ego_box = possible_egos[0]
            else:
                ego_box = min(possible_egos, key=lambda b: 
                    (((b[0] + b[2]) / 2) - self.last_ego_x)**2 + 
                    (((b[1] + b[3]) / 2) - self.last_ego_y)**2
                )
                for b in possible_egos:
                    if not np.array_equal(b, ego_box):
                        obstacles.append(b)
                    
        return ego_box, obstacles, road_boxes

    def calculate_target_x(self, raw_boxes, frame):
        ego_box, obstacles, road_boxes = self.extract_scene_data(raw_boxes)
        
        if ego_box is not None:
            ex1, ey1, ex2, ey2, conf, cls = ego_box
            ego_x = (ex1 + ex2) / 2
            
            self.last_ego_x = ego_x
            self.last_ego_y = (ey1 + ey2) / 2
            self.BOT_Y = ey2 
        else:
            ego_x = self.last_ego_x

        raw_target_x = self.smoothed_target_x 
        raw_left = self.smoothed_left    
        raw_right = self.smoothed_right   
        
        if road_boxes:
            valid_roads = [b for b in road_boxes if b[1] < self.HORIZON_BOTTOM]
            if valid_roads:
                def score_road(b):
                    x1 = max(0, min(self.SCREEN_W, int(b[0])))
                    y1 = max(0, min(self.SCREEN_H, int(b[1])))
                    x2 = max(0, min(self.SCREEN_W, int(b[2])))
                    y2 = max(0, min(self.SCREEN_H, int(b[3])))
                    
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    patch_x1 = max(x1, cx - 10)
                    patch_x2 = min(x2, cx + 10)
                    patch_y1 = max(y1, cy - 10)
                    patch_y2 = min(y2, cy + 10)
                    
                    roi = frame[patch_y1:patch_y2, patch_x1:patch_x2]
                    
                    if roi.size == 0:
                        darkness = 255.0 
                    else:
                        darkness = np.mean(roi)

                    dist_from_path = abs(cx - self.smoothed_target_x)
                    return darkness + (dist_from_path * 0.5)

                main_road = min(valid_roads, key=score_road)
                rx1, ry1, rx2, ry2, _, _ = main_road
                raw_target_x = (rx1 + rx2) / 2
                raw_left = rx1
                raw_right = rx2

        self.smoothed_target_x = (self.alpha * raw_target_x) + ((1 - self.alpha) * self.smoothed_target_x)
        self.smoothed_left = (self.alpha * raw_left) + ((1 - self.alpha) * self.smoothed_left)
        self.smoothed_right = (self.alpha * raw_right) + ((1 - self.alpha) * self.smoothed_right)
        
        # =================================================================
        # FIX 2: PROPORTIONAL DODGING (Based on your Box Size logic!)
        # =================================================================
        total_repulsion = 0
        for obs in obstacles:
            ox1, oy1, ox2, oy2, conf, cls = obs
            obs_cx = (ox1 + ox2) / 2
            
            # The larger the box, the closer the car is to the camera
            box_width = max(1, ox2 - ox1)
            
            dx = ego_x - obs_cx 
            if dx == 0: dx = 1 
            
            # Repulsion scales UP with the car's size, and UP as dx gets smaller
            repulsive_force = (self.K_REP * box_width) / dx
            
            # "PANIC DODGE": If the car is physically in our lane (dx is smaller than the car width)
            # multiply the dodge force by 3 to violently shove the target line away.
            if abs(dx) < (box_width * 1.5): 
                repulsive_force *= 3.0
                
            # Clamp to prevent math explosions, but 400 is strong enough to easily cancel out the road gravity
            repulsive_force = max(-400, min(400, repulsive_force)) 
            total_repulsion += repulsive_force

        dist_left = max(5, ego_x - self.smoothed_left)
        dist_right = max(5, self.smoothed_right - ego_x)
        edge_push_left = self.K_EDGE / (dist_left ** 2)    
        edge_push_right = -self.K_EDGE / (dist_right ** 2) 
        continuous_edge_force = edge_push_left + edge_push_right
            
        wall_repulsion = 0
        if ego_x < (self.smoothed_left + 40): wall_repulsion = (self.smoothed_left + 40 - ego_x) * 6.0 
        elif ego_x > (self.smoothed_right - 40): wall_repulsion = (self.smoothed_right - 40 - ego_x) * 6.0
            
        absolute_safe_x = self.smoothed_target_x + total_repulsion + continuous_edge_force + wall_repulsion
        
        # DODGE TETHER (Anchor to Ego)
        # BUMPED TO 100: The green line can now pull 100 pixels away from the bike, 
        # allowing for wider, more aggressive swerves around big cars.
        MAX_DODGE_WIDTH = 30 
        desired_shift = absolute_safe_x - ego_x
        
        clamped_shift = max(-MAX_DODGE_WIDTH, min(MAX_DODGE_WIDTH, desired_shift))
        calculated_safe_x = ego_x + clamped_shift
        
        # Clamp to Middle 70%
        min_safe_x = self.SCREEN_W * 0.15 
        max_safe_x = self.SCREEN_W * 0.85 
        clamped_safe_x = max(min_safe_x, min(max_safe_x, calculated_safe_x))
        
        # =================================================================
        # FLUID EMA FILTER (Replaces the rigid Slew Rate Limiter)
        # =================================================================
        # A rigid max-shift causes a "sawtooth" wobble when the math spikes.
        # This Exponential Moving Average makes the green line heavy and fluid.
        
        BETA = 0.25  # Moves 25% toward the target each frame. 
        
        final_safe_x = (BETA * clamped_safe_x) + ((1.0 - BETA) * self.last_safe_x)

        # Save state for the next frame
        self.last_safe_x = final_safe_x
        
        return int(final_safe_x), int(ego_x)
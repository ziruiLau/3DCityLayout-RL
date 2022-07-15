import random
import gym
from gym import spaces
import pygame
import numpy as np
from PIL import Image
import copy
import config

# calculate iou
import shapely.geometry
import shapely.affinity
class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

# calculate iou
import shapely.geometry
import shapely.affinity
class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

    def area(self):
        return self.get_contour().area


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 400  # The size of the PyGame window
        self.size = config.SIZE
        self.max_step = config.MAX_STEP
        self.step_count = 0
        self.step_vec = np.zeros((self.max_step), dtype=np.int)
        self.ref_size = 128

        self.ref = np.zeros((1, self.ref_size, self.ref_size))

        self.SCREEN = pygame.display.set_mode((self.window_size, self.window_size))
        self.pix_square_size = (
                self.window_size / self.size
        )

        self.gt = np.zeros((self.size * self.size, 5), dtype=int)
        self.obs = np.zeros((self.size * self.size, 5), dtype=int)

        self.action_num = self.size * self.size * 5 * 2
        self.action_space = spaces.Discrete(self.action_num)
        self._action_to_direction = self.generate_actions_map()

        self.last_IOU = 0
        self.last_local_IOU = 0

        self.window = None
        self.clock = None

    def generate_actions_map(self):
        action_maps = {}
        action = 0
        for i in range(self.size * self.size):
            for j in range(5):
                for k in [-1, 1]:
                    action_maps[action] = np.array([i, j, k])
                    action += 1

        return action_maps

    def valid_mask(self):
        valid_mask = np.ones((self.action_num), dtype=np.int)

        for a in range(self.action_num):

            i, j, k = self._action_to_direction[a]
            box_id = self.step_count % (self.size*self.size)
            # only edit an designated box
            if i != box_id:
                valid_mask[a] = 0

        return valid_mask

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.step_vec = np.zeros((self.max_step), dtype=np.int)
        self.ref = np.zeros((1, self.ref_size, self.ref_size))
        self.last_IOU = 0
        self.last_local_IOU = 0
        self.step_count = 0

        for i in range(self.size*self.size):
            r = i // self.size
            c = i % self.size
            r1 = random.randint(5, int(self.pix_square_size / 3))
            r2 = random.randint(5, int(self.pix_square_size / 3))
            x_t = int(self.pix_square_size * c) + r1
            y_t = int(self.pix_square_size * r) + r2
            self.gt[i][0] = x_t
            self.gt[i][1] = y_t
            self.gt[i][2] = random.randint(int(self.pix_square_size / 2) - 15, int(self.pix_square_size) - r1 - 10) + x_t
            self.gt[i][3] = random.randint(int(self.pix_square_size / 2) - 15, int(self.pix_square_size) - r2 - 10) + y_t
            self.gt[i][4] = random.randint(-10,10)

            self.obs[i][0] = int(self.pix_square_size * c)
            self.obs[i][1] = int(self.pix_square_size * r)
            self.obs[i][2] = int(self.pix_square_size * (c + 1))
            self.obs[i][3] = int(self.pix_square_size * (r + 1))
            self.obs[i][4] = 0

    def reset_for_rl(self):
        self.step_vec = np.zeros((self.max_step), dtype=np.int)
        self.ref_size = 128

        self.ref = np.zeros((1, self.ref_size, self.ref_size))
        self.step_count = 0
        self.last_IOU = 0
        self.last_local_IOU = 0

        for i in range(self.size*self.size):
            r = i // self.size
            c = i % self.size
            self.obs[i][0] = int(self.pix_square_size * c)
            self.obs[i][1] = int(self.pix_square_size * r)
            self.obs[i][2] = int(self.pix_square_size * (c + 1))
            self.obs[i][3] = int(self.pix_square_size * (r + 1))
            self.obs[i][4] = 0

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        box_id = direction[0]
        i = direction[0]
        j = direction[1]
        k = direction[2]

        if j == 4:
            self.obs[i][j] += k
        elif j == 0 | j == 1:
            self.obs[i][j] -= k
        else:
            self.obs[i][j] += k

        IOU, local_IOU = self.compute_increment()
        reward = self.compute_reward(IOU, local_IOU)

        self.last_IOU = IOU
        self.last_local_IOU = local_IOU

        self.step_count += 1
        self.step_vec = np.zeros((self.max_step), dtype=np.int)
        if self.step_count == self.max_step:
            done = True
        else:
            done = False
            self.step_vec[self.step_count] = 1

        return self.obs, self.step_vec, reward, done

    def compute_increment(self):
        i_iou = 0
        u_iou = 0
        single_iou = 0

        for i in range(self.size*self.size):
            gt_b = RotatedRect(self.gt[i][0], self.gt[i][1],
                               self.gt[i][2] - self.gt[i][0],
                               self.gt[i][3] - self.gt[i][1], self.gt[i][4])
            obs_b = RotatedRect(self.obs[i][0], self.obs[i][1],
                               self.obs[i][2] - self.obs[i][0],
                               self.obs[i][3] - self.obs[i][1], self.obs[i][4])

            ins = gt_b.intersection(obs_b).area

            i_iou += ins
            u_iou += gt_b.area() + obs_b.area() - ins
            single_iou += ins / (gt_b.area() + obs_b.area() - ins)

        iou = i_iou / u_iou
        local_iou = single_iou / (self.size * self.size)

        return iou, local_iou

    def compute_reward(self, iou, local_iou):

        r_iou = iou - self.last_IOU
        r_local = local_iou - self.last_local_IOU

        a = 0.1

        reward = r_iou + a * r_local

        return reward

    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        self.SCREEN.fill((255, 255, 255))

        for i in range(self.size * self.size):
            r = i // self.size
            c = i % self.size
            angle = self.gt[i][4]
            surface = pygame.Surface((self.gt[i][2] - self.gt[i][0],
                                      self.gt[i][3] - self.gt[i][1])
                                     , pygame.SRCALPHA)
            surface.fill((255, 0, 0))
            rotated_surface = pygame.transform.rotate(surface, angle)
            rect = rotated_surface.get_rect(center=(
                int((self.gt[i][0] + self.pix_square_size*(c + 1)) / 2),
                int((self.gt[i][1] + self.pix_square_size*(r + 1)) / 2),
            ))
            self.SCREEN.blit(rotated_surface, (rect.x, rect.y))

        for i in range(self.size * self.size):
            r = i // self.size
            c = i % self.size
            angle = self.obs[i][4]

            surface = pygame.Surface((self.obs[i][2] - self.obs[i][0],
                                        self.obs[i][3] - self.obs[i][1],), pygame.SRCALPHA)
            surface.fill((0, 0, 255))
            surface.set_alpha(150)
            rotated_surface = pygame.transform.rotate(surface, angle)
            rect = rotated_surface.get_rect(center=(
                int((self.obs[i][0] + self.pix_square_size * (c + 1)) / 2),
                int((self.obs[i][1] + self.pix_square_size * (r + 1)) / 2),))
            self.SCREEN.blit(rotated_surface, (rect.x, rect.y))

        # self.clock.tick(1)
        pygame.display.update()

    def render_gt_img(self, path):
        self.SCREEN.fill((255, 255, 255))
        for i in range(self.size * self.size):
            r = i // self.size
            c = i % self.size
            angle = self.gt[i][4]
            surface = pygame.Surface((self.gt[i][2] - self.gt[i][0],
                                      self.gt[i][3] - self.gt[i][1])
                                     , pygame.SRCALPHA)
            surface.fill((255, 0, 0))
            rotated_surface = pygame.transform.rotate(surface, angle)
            rect = rotated_surface.get_rect(center=(
                int((self.gt[i][0] + self.pix_square_size * (c + 1)) / 2),
                int((self.gt[i][1] + self.pix_square_size * (r + 1)) / 2),
            ))
            self.SCREEN.blit(rotated_surface, (rect.x, rect.y))

        # pygame.display.update()

        image = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.SCREEN)), axes=(1, 0, 2)
        )
        Image.fromarray(image).save(path + "rgb_array_large.png")

        img = Image.fromarray(image)

        # process and reset reference image
        img = img.convert('L')
        img = img.resize((128, 128), Image.ANTIALIAS)
        raw_img = copy.copy(img)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        ref = img / 255.0
        self.ref = ref
        return ref

    def get_virtual_expert_action_gym(self, valid_mask, random=False):
        max_action = -1
        max_reward = -1000

        for action in range(self.action_num):
            if valid_mask[action] == 0:
                continue

            boxes_, step_, reward, done = self.step_no_update(action)

            if reward > max_reward:
                max_reward = reward
                max_action = action

        return max_action

    def step_no_update(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        box_id = direction[0]
        i = direction[0]
        j = direction[1]
        k = direction[2]

        try_box = self.obs
        if j == 4:
            try_box[i][j] += k
        elif j == 0 | j == 1:
            try_box[i][j] -= k
        else:
            try_box[i][j] += k

        IOU, local_IOU = self.compute_increment()
        reward = self.compute_reward(IOU, local_IOU)

        try_step_vec = np.zeros((self.max_step), dtype=np.int)
        if self.step_count + 1 >= self.max_step:
            done = True
        else:
            done = False
            try_step_vec[self.step_count + 1] = 1



        return self.obs, try_step_vec, reward, done

    def output_result(self, log_info, save_tmp_result_path):
        self.SCREEN.fill((255, 255, 255))

        for i in range(self.size * self.size):
            r = i // self.size
            c = i % self.size
            angle = self.gt[i][4]
            surface = pygame.Surface((self.gt[i][2] - self.gt[i][0],
                                      self.gt[i][3] - self.gt[i][1])
                                     , pygame.SRCALPHA)
            surface.fill((255, 0, 0))
            rotated_surface = pygame.transform.rotate(surface, angle)
            rect = rotated_surface.get_rect(center=(
                int((self.gt[i][0] + self.pix_square_size * (c + 1)) / 2),
                int((self.gt[i][1] + self.pix_square_size * (r + 1)) / 2),
            ))
            self.SCREEN.blit(rotated_surface, (rect.x, rect.y))

        for i in range(self.size * self.size):
            r = i // self.size
            c = i % self.size
            angle = self.obs[i][4]

            surface = pygame.Surface((self.obs[i][2] - self.obs[i][0],
                                      self.obs[i][3] - self.obs[i][1],), pygame.SRCALPHA)
            surface.fill((0, 0, 255))
            surface.set_alpha(150)
            rotated_surface = pygame.transform.rotate(surface, angle)
            rect = rotated_surface.get_rect(center=(
                int((self.obs[i][0] + self.pix_square_size * (c + 1)) / 2),
                int((self.obs[i][1] + self.pix_square_size * (r + 1)) / 2),))
            self.SCREEN.blit(rotated_surface, (rect.x, rect.y))

        # self.clock.tick(1)
        pygame.display.update()

        image = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.SCREEN)), axes=(1, 0, 2)
        )
        Image.fromarray(image).save(save_tmp_result_path + log_info + "-result-t.png")

def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



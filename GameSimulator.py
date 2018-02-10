import itertools as it
from vizdoom import *
import skimage.color
import skimage.transform
import numpy as np

scenario_path = "/home/ghmiao/VizDoomDependFiles/ViZDoom/scenarios/simpler_basic.cfg" # Name and path of scenario

class GameSimulator:
    def __init__(self, frame_repeat=4, resolution=(80, 45, 3)):
        self.game = None
        self.frame_repeat = frame_repeat
        self.resolution = resolution
        self.actions = []
        self.rewards = 0.0
        self.last_action = 0
        
    def initialize(self):
        print("Initializing doom...")
        self.game = DoomGame()
        self.game.load_config(scenario_path)
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_400X225)
        self.game.init()
        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        print(self.actions)
        print(type(self.actions))
        print("Doom initialized.")
    
    # 对游戏图像进行处理，后期可能放到GameSimulator中
    def __preprocess(self, img):
        img = skimage.transform.resize(img, self.resolution, mode='constant')
        img = img.astype(np.float32)
        return img
    
    def get_state(self, preprocess=True):
        # 获取当前游戏的画面，游戏结束则获得空
        if self.game.is_episode_finished():
            return None
        img = self.game.get_state().screen_buffer
        #img = img.reshape([self.screen_height, self.screen_width])
        # 如果进行预处理
        if preprocess: img = self.__preprocess(img)
        return img
    
    def get_action_size(self):
        # 获取动作数目
        return self.game.get_available_buttons_size()
    
    def make_action(self, action):
        reward = self.game.make_action(self.actions[action], self.frame_repeat)
        new_state = self.get_state()
        done = self.is_episode_finished()
        self.rewards += reward
        last_action = self.last_action
        self.last_action = action
        return last_action, new_state, reward, done

    def get_last_action(self):
        return self.last_action
    
    def is_episode_finished(self):
        # 判断游戏是否终止
        return self.game.is_episode_finished()
    
    def reset(self):
        # 重新开始游戏
        self.game.new_episode()
        self.rewards = 0.0
        
    def get_total_reward(self):
        return self.rewards

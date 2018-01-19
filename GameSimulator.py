from ale_python_interface import ale_python_interface

import skimage.color
import skimage.transform
import numpy as np

class GameSimulator:
    def __init__(self, frame_repeat=4, resolution=(80, 45, 3)):
        self.game = None
        self.frame_repeat = frame_repeat
        self.resolution = resolution
        self.actions = []
        self.rewards = 0.0
        self.last_action = 0
        
    def initialize(self, visiable=False, rom=b'pong.bin'):
        # 初始化游戏，返回游戏的动作数目
        print("Initializing Atari...")
        self.game = ale_python_interface.ALEInterface()
        self.game.setBool(b"display_screen", visiable)
        self.game.setBool(b"sound", visiable)
        self.game.loadROM(rom)
        # 获取屏幕大小
        self.screen_width, self.screen_height = self.game.getScreenDims()
        # 初始化动作 one_hot Actions: [0 3 4]
        self.actions = self.game.getMinimalActionSet()
        print('Actions:', self.actions)
        print("Atari initialized.")
        return len(self.actions)
    
    # 对游戏图像进行处理，后期可能放到GameSimulator中
    def __preprocess(self, img):
        img = skimage.transform.resize(img, self.resolution, mode='constant')
        img = img.astype(np.float32)
        return img
    
    def get_state(self, preprocess=True):
        # 获取当前游戏的画面，游戏结束则获得空
        if self.is_terminared():
            return None
        img = self.game.getScreenRGB()
        #img = img.reshape([self.screen_height, self.screen_width])
        # 如果进行预处理
        if preprocess: img = self.__preprocess(img)
        return img
    
    def get_action_size(self):
        # 获取动作数目
        return len(self.game.getMinimalActionSet())
    
    def make_action(self, action):
        # 重复执行动作
        immediate_score = 0.0
        for i in range(self.frame_repeat):
            immediate_score += self.game.act(self.actions[action])
            if self.is_terminared(): break
        reward = immediate_score/np.abs(immediate_score) if immediate_score!=0.0 else 0.0
        new_state = self.get_state()
        done = self.is_terminared()
        self.rewards += reward
        last_action = self.last_action
        self.last_action = action
        return last_action, new_state, reward, done
    
    def is_terminared(self):
        # 判断游戏是否终止
        return self.game.game_over()
    
    def reset(self):
        # 重新开始游戏
        self.game.reset_game()
        self.rewards = 0.0
    
    def close(self):
        # 关闭游戏模拟器
        self.game.close()
        
    def get_total_reward(self):
        return self.rewards

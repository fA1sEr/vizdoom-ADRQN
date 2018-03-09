import itertools as it
import os
from time import time, sleep
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tqdm import trange
from vizdoom import *
from Agent import Agent
from GameSimulator import GameSimulator

# to choose gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

FRAME_REPEAT = 4 # How many frames 1 action should be repeated
UPDATE_FREQUENCY = 4 # How many actions should be taken between each network update
COPY_FREQUENCY = 1000

RESOLUTION = (80, 45, 3) # Resolution
BATCH_SIZE = 32 # Batch size for experience replay
LEARNING_RATE = 0.001 # Learning rate of model
GAMMA = 0.99 # Discount factor

MEMORY_CAP = 200000 # Amount of samples to store in memory

EPSILON_MAX = 1 # Max exploration rate
EPSILON_MIN = 0.1 # Min exploration rate
EPSILON_DECAY_STEPS = 2e5 # How many steps to decay from max exploration to min exploration

RANDOM_WANDER_STEPS = 50000 # How many steps to be sampled randomly before training starts

TRACE_LENGTH = 8 # How many traces are used for network updates
HIDDEN_SIZE = 768 # Size of the third convolutional layer when flattened

EPOCHS = 20000000 # Epochs for training (1 epoch = 200 training Games and 10 test episodes)
GAMES_PER_EPOCH = 1000 # How actions to be taken per epoch
EPISODES_TO_TEST = 10 # How many test episodes to be run per epoch for logging performance
EPISODE_TO_WATCH = 10 # How many episodes to watch after training is complete

TAU = 0.99 # How much the target network should be updated towards the online network at each update

LOAD_MODEL = False # Load a saved model?
SAVE_MODEL = True # Save a model while training?
SKIP_LEARNING = False # Skip training completely and just watch?

model_savefile = "train_data/model.ckpt" # Name and path of the model
reward_savefile = "train_data/Rewards.txt"

##########################################

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def saveScore(score):
    my_file = open(reward_savefile, 'a')  # Name and path of the reward text file
    my_file.write("%s\n" % score)
    my_file.close()

###########################################

game = GameSimulator()
game.initialize()

ACTION_COUNT = game.get_action_size()
print("game.get_action_size()---------------------------")
print(ACTION_COUNT)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

SESSION = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if LOAD_MODEL:
    EPSILON_MAX = 0.25 # restart after 20+ epoch

agent = Agent(memory_cap = MEMORY_CAP, batch_size = BATCH_SIZE, resolution = RESOLUTION, action_count = ACTION_COUNT,
            session = SESSION, lr = LEARNING_RATE, gamma = GAMMA, epsilon_min = EPSILON_MIN, trace_length=TRACE_LENGTH,
            epsilon_decay_steps = EPSILON_DECAY_STEPS, epsilon_max=EPSILON_MAX, hidden_size=HIDDEN_SIZE)

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, TAU)

if LOAD_MODEL:
    print("Loading model from: ", model_savefile)
    saver.restore(SESSION, model_savefile)
else:
    init = tf.global_variables_initializer()
    SESSION.run(init)

##########################################

if not SKIP_LEARNING:
    time_start = time()
    print("\nFilling out replay memory")
    updateTarget(targetOps, SESSION)

    game.reset()
    agent.reset_cell_state()
    state = game.get_state()
    for _ in range(RANDOM_WANDER_STEPS):
        action = agent.random_action()
        last_action, img_state, reward, done = game.make_action(action)
        if not done:
            state_new = img_state
        else:
            state_new = None

        agent.add_transition(last_action, state, action, reward, state_new, done)
        state = state_new

        if done:
            game.reset()
            agent.reset_cell_state()
            state = game.get_state()

    for epoch in range(EPOCHS):
        print("\n\nEpoch %d\n-------" % (epoch))
        print("Training...")

        learning_step = 0
        for games_cnt in range(GAMES_PER_EPOCH):
            game.reset()
            agent.reset_cell_state()
            state = game.get_state()
            while True:
                learning_step += 1
                action = agent.act(game.get_last_action(), state)
                last_action, img_state, reward, done = game.make_action(action)
                if not done:
                    state_new = img_state
                else:
                    state_new = None
                agent.add_transition(last_action, state, action, reward, state_new, done)
                state = state_new

                if learning_step % UPDATE_FREQUENCY == 0:
                    agent.learn_from_memory()
                if learning_step % COPY_FREQUENCY == 0:
                    updateTarget(targetOps, SESSION)

                if done:
                    print("Epoch %d Train Game %d get %.1f" % (epoch, games_cnt, game.get_total_reward()))
                    break
            if SAVE_MODEL and games_cnt % 10 == 0:
                saver.save(SESSION, model_savefile)
                print("Saving the network weigths to:", model_savefile)

        print("\nTesting...")

        test_scores = []
        for test_step in range(EPISODES_TO_TEST):
            game.reset()
            agent.reset_cell_state()
            while not game.is_episode_finished():
                state = game.get_state()
                action = agent.act(game.get_last_action(), state, train=False)
                game.make_action(action)
            test_scores.append(game.get_total_reward())

        test_scores = np.array(test_scores)
        print("Results: mean: %.1f±%.1f," % (test_scores.mean(), test_scores.std()),
              "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

        if SAVE_MODEL:
            saveScore(test_scores.mean())
            saver.save(SESSION, model_savefile)
            print("Saving the network weigths to:", model_savefile)
            if epoch % (EPOCHS/5) == 0 and epoch is not 0:
                saver.save(SESSION, model_savefile, global_step=epoch)

        print("Total ellapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
'''
print("TIME TO WATCH!!")
# Reinitialize the game with window visible
game.close()
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()
score = []
for _ in trange(EPISODE_TO_WATCH, leave=False):
    game.new_episode()
    agent.reset_cell_state()
    while not game.is_episode_finished():
        state = preprocess(game.get_state().screen_buffer)
        action = agent.act(state, train=False)
        game.set_action(actions[action])
        for i in range(FRAME_REPEAT):
            game.advance_action()
            done = game.is_episode_finished()
            if done:
                break

    # Sleep between episodes
    sleep(1.0)
    score.append(game.get_total_reward())
score = np.array(score)
game.close()
print("Results: mean: %.1f±%.1f," % (score.mean(), score.std()),
          "min: %.1f" % score.min(), "max: %.1f" % score.max())
'''

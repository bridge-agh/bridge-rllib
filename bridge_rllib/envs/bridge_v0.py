import numpy as np
import gymnasium as gym
from gymnasium import spaces

import bridge_core_py.core as bc


ALL_CARDS = [bc.Card(suit, rank) for suit in bc.Suit for rank in bc.Rank]
CARD_TO_INT = {card: i for i, card in enumerate(ALL_CARDS)}

ALL_ACTIONS = ALL_CARDS
ACTION_TO_INT = {action: i for i, action in enumerate(ALL_ACTIONS)}

TRICK_REWARD = 1 / 13


class BridgeEnv(gym.Env):
    def __init__(self, reward_mode='sparse', render_mode=None):
        assert reward_mode in ('sparse', 'shaped')
        self._reward_mode = reward_mode
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ALL_ACTIONS))
        observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(sum((
                4, # current player
                4, # dummy
                52, # dummy cards
                52, # hand
                52, # played cards
            )),))
        action_mask = spaces.Box(
            low=0, high=1, shape=(len(ALL_ACTIONS),), dtype=np.int8)
        self.observation_space = spaces.Dict({
            'action_mask': action_mask,
            'observation': observation_space,
        })

    def step(self, action):
        reward = 0.0

        our_tricks = self._get_pair_tricks(self._we_are)
        their_tricks = self._get_pair_tricks(self._they_are)

        action = ALL_ACTIONS[action]
        try:
            self._bridge.step(action)
        except ValueError:
            raise ValueError('You shall not pass!')
            reward = -1.0
            return self._get_obs(), reward, True, False, {}
        self._played_cards.append(action)

        if self._get_pair_tricks(self._we_are) > our_tricks:
            reward += TRICK_REWARD
        if self._get_pair_tricks(self._they_are) > their_tricks:
            reward -= TRICK_REWARD

        if self._bridge.stage == bc.GameStage.SCORING:
            if self._reward_mode == 'sparse':
                if self._get_pair_tricks(self._we_are) > self._get_pair_tricks(self._they_are):
                    reward = 1.0
                elif self._get_pair_tricks(self._they_are) > self._get_pair_tricks(self._we_are):
                    reward = -1.0
                else:
                    reward = 0.0
            return self._get_obs(), reward, True, False, {}

        while not self._is_our_turn():
            our_tricks = self._get_pair_tricks(self._we_are)
            their_tricks = self._get_pair_tricks(self._they_are)

            action = self.np_random.choice(self._bridge.actions())
            self._bridge.step(action)
            self._played_cards.append(action)

            if self._get_pair_tricks(self._we_are) > our_tricks:
                reward += TRICK_REWARD
            elif self._get_pair_tricks(self._they_are) > their_tricks:
                reward -= TRICK_REWARD

            if self._bridge.stage == bc.GameStage.SCORING:
                if self._reward_mode == 'sparse':
                    if self._get_pair_tricks(self._we_are) > self._get_pair_tricks(self._they_are):
                        reward = 1.0
                    elif self._get_pair_tricks(self._they_are) > self._get_pair_tricks(self._we_are):
                        reward = -1.0
                    else:
                        reward = 0.0
                return self._get_obs(), reward, True, False, {}

        if self._reward_mode == 'sparse':
            reward = 0

        return self._get_obs(), reward, False, False, {}

    def _is_our_turn(self):
        if self._we_are == 'NS':
            return self._bridge.current_player in (bc.PlayerDirection.NORTH, bc.PlayerDirection.SOUTH)
        else:
            return self._bridge.current_player in (bc.PlayerDirection.EAST, bc.PlayerDirection.WEST)

    def _get_pair_tricks(self, pair):
        if pair == 'NS':
            return len(self._bridge.NS_tricks)
        else:
            return len(self._bridge.EW_tricks)

    def action_masks(self):
        mask = np.zeros(len(ALL_ACTIONS),
                        dtype=self.observation_space['action_mask'].dtype)
        for action in self._bridge.actions() or []:
            mask[ACTION_TO_INT[action]] = 1
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            try:
                self._bridge = bc.Game(seed=seed or self.np_random.integers(0, 2**32))
                self._played_cards = []
                while self._bridge.stage != bc.GameStage.PLAYING:
                    self._bridge.step(self.np_random.choice(self._bridge.actions()))
                break
            except:
                continue
        if self._bridge.current_player in (bc.PlayerDirection.NORTH, bc.PlayerDirection.SOUTH):
            self._we_are = 'NS'
            self._they_are = 'EW'
        else:
            self._we_are = 'EW'
            self._they_are = 'NS'
        if self.np_random.random() < 0.5:
            action = self.np_random.choice(self._bridge.actions())
            self._bridge.step(action)
            self._played_cards.append(action)
            assert not self._is_our_turn()
            self._we_are, self._they_are = self._they_are, self._we_are
        assert self._bridge.stage == bc.GameStage.PLAYING
        return self._get_obs(), {}

    def _get_obs(self):
        dtype = self.observation_space['observation'].dtype
        obs_current_player = np.zeros(4, dtype=dtype)
        obs_dummy = np.zeros(4, dtype=dtype)
        obs_dummy_cards = np.zeros(52, dtype=dtype)
        obs_hand = np.zeros(52, dtype=dtype)
        obs_played_cards = np.zeros(52, dtype=dtype)

        bridge_obs = self._bridge.player_observation(self._bridge.current_player)

        # current player
        obs_current_player[bridge_obs['current_player'].value] = 1

        # dummy
        if len(bridge_obs['game']['dummy']) > 0:
            declarer = bridge_obs['bidding']['declarer'].value
            obs_dummy[declarer] = 1
            for card in bridge_obs['game']['dummy']:
                obs_dummy_cards[CARD_TO_INT[card]] = 1

        # hand
        for card in bridge_obs['hand']:
            obs_hand[CARD_TO_INT[card]] = 1

        # played cards
        for card in self._played_cards:
            obs_played_cards[CARD_TO_INT[card]] = 1

        obs = np.concatenate((
            obs_current_player.flatten(),
            obs_dummy.flatten(),
            obs_dummy_cards.flatten(),
            obs_hand.flatten(),
            obs_played_cards.flatten(),
        ))

        mask = self.action_masks()

        return {
            'action_mask': mask,
            'observation': obs,
        }

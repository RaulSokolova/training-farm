from __future__ import annotations
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='/home/theimmigrant/FinRLOptimized/trading_log.txt',  # Name of the log file
                    filemode='w')  # 'w' to overwrite each time, 'a' to append


import gymnasium as gym
import numpy as np
from numpy import random as rd

HIGH_PROFIT_LIMIT = 0.05 # 2% 
STOPLOSS_LIMIT = 0.01 # 1% limit
# PENALTIES AND REWARD
HIGH_PROFIT_RW = 1.03
STOPLOSS_PENALTY = 1.04
# 
CASH_PENALTY_PROPORTION = 0.1


class StockTradingEnvV3(gym.Env):
    def __init__(
        self,
        config,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=1e-4,
        initial_stocks=None,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.stocks_avg_buy_price = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.trailing_stop_loss = np.full(stock_dim, -np.inf)

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )




    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_avg_buy_price = (
                self.initial_stocks + rd.randint(100, 200, size=self.initial_stocks.shape)
            ).astype(np.float32)

            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_avg_buy_price = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price), {}  # state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1
        profit_losses= np.zeros(4)

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                result = self._sell(price, index, actions)
                profit_losses += result

            for index in np.where(actions > min_action)[0]:  # buy_index:
                self._buy(price, index, actions)
        else:  # sell all when turbulence
            self._sell_all(price)

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        self.total_asset = total_asset
        # reward = (total_asset - self.total_asset) * self.reward_scaling
        reward= self.get_reward(profit_losses)
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step

        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, False, dict()

    def _sell(self, price, index, actions):
        logging.debug(f"Selling - Day: {self.day}, Price: {price[index]}, Index: {index}, Actions: {actions}")
        low_profit = 0
        profit = 0
        loss = 0
        stop_losses = 0
        perc_diff = np.nan  # Initialize perc_diff

        if price[index] > 0:  # Sell only if current asset is > 0
            sell_num_shares = min(self.stocks[index], -actions[index])
            sell_amount = price[index] * sell_num_shares * (1 - self.sell_cost_pct)
            bought_amount = self.stocks_avg_buy_price[index] * sell_num_shares * (1 + self.buy_cost_pct)

            # Evaluating trade results
            diff_sell_buy_amount = sell_amount - bought_amount
            if bought_amount != 0 and not np.isnan(bought_amount):
                perc_diff = diff_sell_buy_amount / bought_amount
                logging.debug(f"Percentage Difference: {perc_diff}")

            # Check for dynamic and trailing stop loss
            if self.stocks_avg_buy_price[index] != 0 and not np.isnan(self.stocks_avg_buy_price[index]):
                current_loss = (price[index] - self.stocks_avg_buy_price[index]) / self.stocks_avg_buy_price[index]
                dynamic_stop_loss = self.calculate_dynamic_stop_loss(index)

                if current_loss <= dynamic_stop_loss or current_loss <= self.trailing_stop_loss[index]:
                    # Implement stop loss selling
                    stop_losses = diff_sell_buy_amount
                    # Update the sell_num_shares or total sell_amount as per your strategy

        # Classify the selling outcome
        if not np.isnan(perc_diff):
            if perc_diff >= HIGH_PROFIT_LIMIT:
                profit = diff_sell_buy_amount
            elif perc_diff >= 0:
                low_profit = diff_sell_buy_amount
            else:
                loss = diff_sell_buy_amount

        self.stocks[index] -= sell_num_shares
        self.amount += sell_amount
        self.stocks_cool_down[index] = 0

        return np.array([low_profit, profit, loss, stop_losses])


    def _sell_all(self, price):
        self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
        self.stocks[:] = 0
        self.stocks_avg_buy_price[:] = 0
        self.stocks_cool_down[:] = 0

    def _buy(self, price, index, actions):
        logging.debug(f"Buying - Day: {self.day}, Price: {price[index]}, Index: {index}, Actions: {actions}")
        if price[index] > 0:  # Buy only if the price is > 0
            max_affordable_shares = self.amount // price[index]
            buy_num_shares = min(max_affordable_shares, actions[index])
            if buy_num_shares > 0:
                total_buy_cost = price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                if self.amount >= total_buy_cost:
                    self.stocks_avg_buy_price[index] = ((self.stocks[index] * self.stocks_avg_buy_price[index]) + (buy_num_shares * price[index])) / (self.stocks[index] + buy_num_shares)
                    self.stocks[index] += buy_num_shares
                    self.amount -= total_buy_cost
                    self.stocks_cool_down[index] = 0
                    self.update_trailing_stop_loss(index, price[index])
                    logging.debug(f"Bought {buy_num_shares} shares of stock {index} at price {price[index]}")
                else:
                    logging.debug(f"Insufficient funds to buy stock {index}")


    def get_state(self, price):
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    def get_reward(self,profit_losses):
        if self.day == 0:
            return 0
        else:

            holding_indices = np.nonzero(self.stocks)[0]

            # CASH PENALTY / Penalty if cash levels are to low
            cash_penalty = max(0, (self.total_asset * CASH_PENALTY_PROPORTION - self.amount))
            
            
            # STOP LOSS PENALTY
            stop_losses_penalty = profit_losses[3] * STOPLOSS_PENALTY

            #  LOSS PENALTY
            losses_penalty = profit_losses[2]

            # # TOTAL PENALTY
            total_penalty = cash_penalty + stop_losses_penalty + losses_penalty


            # LOW PROFIT REWARD
            low_profit_reward = profit_losses[0]

            # HIGH PROFIT REWARD
            profit_reward = profit_losses[1] * HIGH_PROFIT_RW

            # TOTAL REWARD
            total_reward = low_profit_reward + profit_reward

            # REWARD
            reward = (total_reward - total_penalty ) * self.reward_scaling

            # Example risk adjustment factor based on volatility or other measures
            # Calculate risk adjustment factor
            portfolio_volatility = self.calculate_portfolio_volatility()
            risk_adjustment_factor = 1 / (1 + portfolio_volatility)  # Example: Inverse of volatility

            # Adjust reward based on risk
            reward = (total_reward - total_penalty) * self.reward_scaling * risk_adjustment_factor
            return reward

    def calculate_dynamic_stop_loss(self, index):
        if self.day > 0:
            historical_volatility = np.std(self.price_ary[:self.day, index])
            return -0.01 - historical_volatility
        else:
            return 0  # Default value, or handle this scenario appropriately
    def update_trailing_stop_loss(self, index, current_price):
        if not np.isnan(current_price):
            self.trailing_stop_loss[index] = max(self.trailing_stop_loss[index], current_price - 0.05 * current_price)

    def calculate_portfolio_volatility(self):
        if self.day > 1:  # Needs at least 2 days for a meaningful calculation
            historical_price_changes = np.diff(self.price_ary[:self.day], axis=0)
            portfolio_volatility = np.std(historical_price_changes)
            return portfolio_volatility
        else:
            return 0 

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

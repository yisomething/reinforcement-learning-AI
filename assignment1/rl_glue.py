"""
Glues together an experiment, agent, and environment.
"""
from abc import ABCMeta, abstractmethod


class RLGlue:
    """
    Facilitates interaction between an agent and environment for
    reinforcement learning experiments.

    args:
        env_obj: an object that implements BaseEnvironment
        agent_obj: an object that implements BaseAgent
    """

    def __init__(self, env_obj, agent_obj):
        self._environment = env_obj
        self._agent = agent_obj

        # useful statistics
        self._total_reward = None
        self._num_steps = None  # number of steps in entire experiment
        self._num_episodes = None  # number of episodes in entire experiment
        self._num_ep_steps = None  # number of steps in this episode

        # the most recent action taken by the agent
        self._last_action = None

    def total_reward(self):
        return self._total_reward

    def num_steps(self):
        return self._num_steps

    def num_episodes(self):
        return self._num_episodes

    def num_ep_steps(self):
        return self._num_ep_steps

    def rl_init(self):
        # reset statistics
        self._total_reward = 0
        self._num_steps = 0
        self._num_episodes = 0
        self._num_ep_steps = 0

        # reset last action
        self._last_action = None

        # reset agent and environment
        self._agent.agent_init()
        self._environment.env_init()

    def rl_start(self):
        """
        Starts RLGlue experiment.

        Returns:
            tuple: (state, action)
        """
        self._num_ep_steps = 1
        self._num_steps = max(self._num_steps, 1)

        state = self._environment.env_start()
        self._last_action = self._agent.agent_start(state)

        return state, self._last_action

    def rl_step(self):
        """Takes a step in the RLGlue experiment.

        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """
        reward, state, terminal = self._environment.env_step(self._last_action)

        self._total_reward += reward

        if terminal:
            self._num_episodes += 1
            self._agent.agent_end(reward)
            self._last_action = None
        else:
            self._num_ep_steps += 1
            self._num_steps += 1
            self._last_action = self._agent.agent_step(reward, state)

        return reward, state, self._last_action, terminal

    ### CONVENIENCE FUNCTIONS BELOW ###
    def rl_env_start(self):
        """
        Useful when manually specifying agent actions (for debugging). Starts
        RL-Glue environment.

        Returns:
            state observation
        """
        self._num_ep_steps = 0

        return self._environment.env_start()

    def rl_env_step(self, action):
        """
        Useful when manually specifying agent actions (for debugging).Takes a
        step in the environment based on an action.

        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        reward, state, terminal = self._environment.env_step(action)

        self._total_reward += reward

        if terminal:
            self._num_episodes += 1
        else:
            self._num_ep_steps += 1
            self._num_steps += 1

        return reward, state, terminal

    def rl_episode(self, max_steps_this_episode=0):
        """
        Convenience function to run an episode.

        Args:
            max_steps_this_episode (Int): Max number of steps in this episode.
                A value of 0 will result in the episode running until
                completion.

        returns:
            Boolean: True if the episode terminated within
                max_steps_this_episode steps, else False
        """
        terminal = False

        self.rl_start()

        while not terminal and ((max_steps_this_episode <= 0) or
                                (self._num_ep_steps < max_steps_this_episode)):
            _, _, _, terminal = self.rl_step()

        return terminal

    def rl_agent_message(self, message):
        """
        pass a message to the agent

        Args:
            message (str): the message to pass

        returns:
            str: the agent's response
        """
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_agent_response = self._agent.agent_message(message_to_send)
        if the_agent_response is None:
            the_agent_response = ""

        return the_agent_response

    def rl_env_message(self, message):
        """
        pass a message to the environment

        Args:
            message (str): the message to pass

        Returns:
            the_env_response (str) : the environment's response
        """
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_env_response = self._environment.env_message(message_to_send)
        if the_env_response is None:
            return ""

        return the_env_response


class BaseAgent:
    """
    Defines the interface of an RLGlue Agent

    ie. These methods must be defined in your own Agent classes
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Declare agent variables."""
        pass

    @abstractmethod
    def agent_init(self):
        """Initialize agent variables."""

    @abstractmethod
    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, state):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (state observation): The agent's current state
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

    @abstractmethod
    def agent_message(self, message):
        """
        receive a message from rlglue
        args:
            message (str): the message passed
        returns:
            str : the agent's response to the message (optional)
        """



class BaseEnvironment:
    """
    Defines the interface of an RLGlue environment

    ie. These methods must be defined in your own environment classes
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Declare environment variables."""

    @abstractmethod
    def env_init(self):
        """
        Initialize environment variables.
        """

    @abstractmethod
    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def env_step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

    @abstractmethod
    def env_message(self, message):
        """
        receive a message from RLGlue
        Args:
           message (str): the message passed
        Returns:
           str: the environment's response to the message (optional)
        """

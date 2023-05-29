from hex.adversary.base_adversary import BaseAdversary


class LeagueAdversary(BaseAdversary):

    def __init__(self):
        # TODO: Add configuration parameters, if needed, or path to saved networks
        super().__init__()
        self.current_network = None

    def init(self, q_learner):
        # TODO: Create networks with q_learner.model.make_network(), or load them from disk
        pass

    def get_action(self, state, q_learner):
        # TODO: Return an action for the given state, with the current network
        pass

    def update(self, q_learner, epoch):
        # Called every epoch (before every game)
        # TODO: Network switching logic
        pass

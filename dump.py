My randomsstart is having an error. I think its somehow not recorded on the board or smth similiar. 

Code.

    def learn(self, num_episodes=500,
              batch_size=100,
              gamma=0.99,
              eps_start=0.9,
              eps_end=0.01,
              eps_decay=1000,
              soft_update=True,
              target_net_update_rate=0.005,
              learning_rate=1e-4,
              eval_every=10,
              save_every=100,
              start_from_model=None,
              random_start=False,
              save_path="models/model.pt",
              evaluate_runs=100,
              clip_grads=100,
              playAsColor=0.5):

        if start_from_model is not None:
            if os.path.isfile(start_from_model):
                self.model.load_model(start_from_model)

        def eps_by_step(step):
            return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)

        optimizer = optim.SGD(self.model.policy_net.parameters(), lr=learning_rate, momentum=0.9)

        steps_done = 0

        winners = []
        for i_episode in range(num_episodes):
            self.adversary.update(self, epoch=i_episode, random_start=random_start, showPlot=True)

            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            play_as_white = random.random() > playAsColor
            if random_start:
                if play_as_white:
                    action = self._eps_greedy_action(state, eps=2)
                    observation, reward, terminated, next_action_space = self.env.step(action.item())
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
                    action = self.adversary.get_action(state, self)
                    observation, reward, terminated, next_action_space = self.env.step(action.item())
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    reward_t = torch.tensor([reward if play_as_white else -reward], device=self.device)
                    self.memory.save(state, action, next_state, reward_t, next_action_space)

                else:
                    action = self.adversary.get_action(state, self)
                    observation, reward, terminated, next_action_space = self.env.step(action.item())
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    action = self._eps_greedy_action(state, eps=2)
                    observation, reward, terminated, next_action_space = self.env.step(action.item())
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
                    action = self.adversary.get_action(state, self)
                    observation, reward, terminated, next_action_space = self.env.step(action.item())
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    reward_t = torch.tensor([reward if play_as_white else -reward], device=self.device)
                    self.memory.save(state, action, next_state, reward_t, next_action_space)

                    
           
            if(not play_as_white and not random_start):
                action = self.adversary.get_action(state, self)
                state, _, _, _ = self.env.step(action.item())
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)


            for t in count():
                steps_done += 1

                action = self._eps_greedy_action(state, eps=eps_by_step(steps_done))
                observation, reward, terminated, next_action_space = self.env.step(action.item())

                taken_action = action
                reward_t = torch.tensor([reward if play_as_white else -reward], device=self.device)
                done = terminated
                observation2 = observation
                next_action_space2 = None
                if not terminated:
                    action = self.adversary.get_action(
                        torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0), self)
                    observation2, reward2, terminated2, next_action_space2 = self.env.step(action.item())

                    if terminated2:
                        reward_t = torch.tensor([reward2 if play_as_white else -reward2], device=self.device)
                        done = terminated2

                if not done:
                    next_state = torch.tensor(observation2, dtype=torch.float32, device=self.device).unsqueeze(0)
                    mask = [-2] * self.n_actions
                    for i in next_action_space2:
                        mask[i] = 1
                    next_action_space = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    next_state = None
                    next_action_space = None

                self.memory.save(state, taken_action, next_state, reward_t, next_action_space)

                state = next_state

                self.optimize_model(optimizer, batch_size, gamma, clip_grads=clip_grads)

                if soft_update:
                    for target_param, param in zip(self.model.target_net.parameters(), self.model.policy_net.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - target_net_update_rate) +
                                                param.data * target_net_update_rate)
                        
                else:
                    if steps_done % target_net_update_rate == 0:
                        self.model.target_net.load_state_dict(self.model.policy_net.state_dict())

                if done:
                    self.train_reward_history.append(reward_t.item())
                    break
                
            if(play_as_white):
                if(self.env.engine.winner == 1):
                    winners.append(1)
                else:
                    winners.append(-1)
            else:
                if(self.env.engine.winner == -1):
                    winners.append(1)
                else:
                    winners.append(-1)
            if i_episode % (eval_every/4) == 0:
                print("Self Wins: ", winners.count(1), "Adv Wins: ", winners.count(-1))

            if i_episode % (eval_every) == 0:
                    winners.clear()
            if i_episode % save_every == 0:
                torch.save(self.model.policy_net.state_dict(), save_path)
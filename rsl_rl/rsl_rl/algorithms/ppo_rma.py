import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage import RolloutStorage

class PPORMA:
    actor_crtic: ActorCriticRMA
    def __init__(self,
                 actor_critic,
                 estimator,
                 estimator_params,
                 depth_encoder,
                 depth_encoder_paras,
                 depth_actor,
                 device='cpu',
                 # algorithm cfg
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 priv_reg_coef_schedual = [0, 0, 0],
                 **kwargs
                 ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        
        # ppo components
        self.actor_crtic = actor_critic
        self.actor_crtic.to(self.device)
        self.storage = None
        self.AC_optimizer = optim.Adam(self.actor_crtic.parameters(), lr=self.learning_rate)
        # print(self.actor_crtic.parameters())
        self.transition = RolloutStorage.Transition()
        
        # ppo parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        # adaption                                      #? 这个parameters和上面的parameters有什么区别。再看下parkour 网络结构的代码
        self.hist_encoder_optimizer = optim.Adam(self.actor_crtic.actor.history_encoder.parameters(), lr=self.learning_rate)
        # print(self.actor_crtic.actor.history_encoder.parameters())
        self.priv_reg_coef_schedual = priv_reg_coef_schedual    # [0, 0.1, 2000, 3000]
        self.counter = 0
        
        # estimator 
        self.estimator = estimator
        self.priv_states_dim = estimator_params["priv_states_dim"]
        self.num_prop = estimator_params["num_prop"]
        # self.num_scandots = estimator_params["num_scan"]
        self.num_scandots_terrain = estimator_params["num_scan_terrain"]
        self.num_scandots_ceiling = estimator_params["num_scan_ceiling"]
        
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_params["learning_rate"])
        self.train_with_estimated_states = estimator_params["train_with_estimated_states"]

        # depth encoder
        self.if_depth = depth_encoder!=None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"])
            self.depth_encoder_paras = depth_encoder_paras
            
            self.depth_actor = depth_actor
            #! 同时优化depth_encoder和depth_actor的参数
            self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, 
                                      num_transitions_per_env, 
                                      actor_obs_shape, 
                                      critic_obs_shape, 
                                      action_shape,
                                      self.device)

    def test_mode(self):
        self.actor_crtic.eval()
        
    def train_mode(self):
        self.actor_crtic.train()
    
    '''
    Sample actions from action distribution. Choose to use estimated states or not.
    '''
    def act(self, obs, critic_obs, hist_encoding=False):
        
        if self.actor_crtic.is_recurrent:   # false
            self.transition.hidden_states = self.actor_crtic.get_hidden_states()

        if self.train_with_estimated_states:
            obs_est = obs.clone()
            
            # 通过本体的proprioception 估计priv_explicit(本体速度还有质心高度)
            priv_state_estimated = self.estimator(obs_est[: , :self.num_prop])  # [num_env, 4]
            
            # 把原本的priv_explicit替换成估计的priv_explicit
            obs_est[:, self.num_prop+self.num_scandots_terrain+self.num_scandots_ceiling:self.num_prop+self.num_scandots_terrain+self.num_scandots_ceiling+self.priv_states_dim] = priv_state_estimated
            
            self.transition.actions = self.actor_crtic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_crtic.act(obs, hist_encoding).detach()
            
        self.transition.values = self.actor_crtic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_crtic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_crtic.get_action_mean.detach()
        self.transition.action_sigma = self.actor_crtic.get_action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        return self.transition.actions

    '''
    Save transitions to storage
    '''
    def process_env_step(self, rewards, dones, infos):
        rewards_total = rewards.clone()

        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        
        # 对time_)outs进行特殊处理
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_crtic.reset(dones)   # not implemented
        
        return rewards_total
    
    '''
    compute  advantage and return
    '''
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_crtic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_priv_reg_loss = 0
        
        if self.actor_crtic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs)
            
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch_none, masks_batch_none in generator:
                
            # self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            self.actor_crtic.act(obs_batch) # update action distribution
            
            actions_log_prob_batch = self.actor_crtic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_crtic.evaluate(critic_obs_batch)
            mu_batch = self.actor_crtic.get_action_mean
            sigma_batch = self.actor_crtic.get_action_std
            entropy_batch = self.actor_crtic.get_action_entropy
            
            #! history adaption module update： 更新使用state history encoder预测priv env latent
            priv_envinfo_latent_batch = self.actor_crtic.actor.infer_priv_envinfo_latent(obs_batch)
            with torch.inference_mode():
                estimated_envinfo_batch = self.actor_crtic.actor.infer_hist_latent(obs_batch)
            privinfo_reg_loss = (priv_envinfo_latent_batch - estimated_envinfo_batch.detach()).norm(p=2, dim=1).mean()
            #  MIN[ max[(counter - 2000),0] / 3000, 1 ]
            priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3] , 1)
            # stage * 0.1
            priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]
            
            #! estimator update: 更新使用当前state 预测priv self latent
            estimated_selfinfo_batch = self.estimator(obs_batch[:, :self.num_prop])
            estimator_loss = (estimated_selfinfo_batch - obs_batch[:,self.num_prop+self.num_scandots_terrain+self.num_scandots_ceiling:self.num_prop+self.num_scandots_terrain+self.num_scandots_ceiling+self.priv_states_dim]).pow(2).mean() 
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()
            
            # adaptive learning rate
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                    for param_group in self.AC_optimizer.param_groups:
                        param_group['lr'] = self.learning_rate  # 修改optimizer參數
                        
            # surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            
            # value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch-target_values_batch).clamp(-self.clip_param,self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + \
                    self.value_loss_coef * value_loss - \
                    self.entropy_coef * entropy_batch.mean() + \
                    priv_reg_coef * privinfo_reg_loss   # history adaption module loss
                   
            #! actor critic update 
            self.AC_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_crtic.parameters(), self.max_grad_norm) # history_encoder的Params包含在actor_critic中
            self.AC_optimizer.step()    #! 这边优化的到底是哪些参数
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_priv_reg_loss += privinfo_reg_loss.item()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches  
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        
        self.storage.clear()
        self.update_counter()
        
        return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_priv_reg_loss, 0, 0, priv_reg_coef
            
            
    def update_counter(self):
        self.counter += 1
        
    '''
    update history encoder 
    '''
    def update_dagger(self):
        mean_hist_latent_loss = 0
        
        if self.actor_crtic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            
            with torch.inference_mode():
                self.actor_crtic.act(obs_batch, use_historyestimate=True)   # update action distributiond
                
            with torch.inference_mode():
                privinfo_latent_batch = self.actor_crtic.actor.infer_priv_envinfo_latent(obs_batch)
                
            privinfo_hist_latent_batch = self.actor_crtic.actor.infer_hist_latent(obs_batch)
            
            hist_latent_loss = (privinfo_latent_batch.detach()-privinfo_hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_crtic.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            
            mean_hist_latent_loss += hist_latent_loss.item()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        
        self.storage.clear()
        self.update_counter()
        
        return mean_hist_latent_loss
    
    '''
    update depth encoder
    '''
    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        if self.depth_encoder != None:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
            
            self.depth_encoder_optimizer.zero_grad()
            depth_encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
            self.depth_encoder_optimizer.step()
            
            return depth_encoder_loss.item()
        
    '''
    update depth actor
    '''
    def update_depth_actor(self, actions_student_batch, actions_teacher_batch):
        if self.depth_actor != None:
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
            
            self.depth_actor_optimizer.zero_grad()
            depth_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
            self.depth_actor_optimizer.step()
            
            return depth_actor_loss.item()
        
        
    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
        if self.depth_encoder != None:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
            
            depth_loss = depth_encoder_loss + depth_actor_loss
            
            self.depth_actor_optimizer.zero_grad()
            depth_loss.backward()
            nn.utils.clip_grad_norm_([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], self.max_grad_norm)
            self.depth_actor_optimizer.step()
            
            return depth_encoder_loss.item(), depth_actor_loss.item()
        
        
    
    
        

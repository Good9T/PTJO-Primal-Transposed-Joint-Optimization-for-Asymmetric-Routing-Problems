import torch
from logging import getLogger


from ACVRPEnv import ACVRPEnv as Env
from ACVRPModel import ACVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils import *

class ACVRPTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(device)
        else:
            device = torch.device('cpu')
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(device)

        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pth'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded!')

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        epochs = self.trainer_params['epochs']
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info('-------------------------------------------------')
            train_score, train_loss, loss1, loss2= self._train_one_epoch(epoch)
            self.scheduler.step()
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('train_loss_primal', epoch, loss1)
            self.result_log.append('train_loss_transposed', epoch, loss2)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, epochs)
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], remain[{}]".format(epoch, epochs, elapsed_time_str, remain_time_str))

            all_done = (epoch == epochs)
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:
                self.logger.info('Save log_image')
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss_primal'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss_transposed'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info('Save model')
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss_primal'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss_transposed'])

            if all_done:
                self.logger.info('All done!')
                self.logger.info('Print log')
                util_print_log_array(self.logger, self.result_log)


    def _train_one_epoch(self, epoch):
        score = AverageMeter()
        loss = AverageMeter()
        loss1 = AverageMeter()
        loss2 = AverageMeter()
        train_episode_num = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_episode_num:
            remaining  = train_episode_num - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            avg_score, avg_loss, avg_loss1, avg_loss2 = self._train_one_batch(batch_size)
            score.update(avg_score, batch_size)
            loss.update(avg_loss, batch_size)
            loss1.update(avg_loss1, batch_size)
            loss2.update(avg_loss2, batch_size)
            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%) Score: {:.4f}, Loss: {:.4f}, Loss_primal: {:.4f}, Loss_transposed: {:.4f}'
                                     .format(epoch, episode, train_episode_num, 100. * episode / train_episode_num,
                                             score.avg, loss.avg, loss1.avg, loss2.avg))

        self.logger.info('Epoch {:3d}: Train ({:3.0f}%) Score: {:.4f}, Loss: {:.4f}, Loss_primal: {:.4f}, Loss_transposed: {:.4f}'
                         .format(epoch, 100. * episode / train_episode_num, score.avg, loss.avg, loss1.avg, loss2.avg))

        return score.avg, loss.avg, loss1.avg, loss2.avg


    def _train_one_batch(self, batch_size):
        self.model.train()
        # primal problem
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)
        prob_list = torch.zeros(batch_size, self.env.pomo_size, 0)
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        primal_log_prob = prob_list.log().sum(dim=2)

        # transposed problem
        self.env.load_transposed_problems()
        t_reset_state, _, _ = self.env.reset()
        self.model.pre_forward(t_reset_state)
        t_prob_list = torch.zeros(batch_size, self.env.pomo_size, 0)
        t_state, t_reward, t_done = self.env.pre_step()
        while not t_done:
            t_selected, t_prob = self.model(t_state)
            t_state, t_reward, t_done = self.env.step(t_selected)
            t_prob_list = torch.cat((t_prob_list, t_prob[:, :, None]), dim=2)

        trans_log_prob = t_prob_list.log().sum(dim=2)
        combined_reward = torch.cat([reward, t_reward], dim=1)
        # (batch , 2*pomo)

        # advantage
        combined_reward_float = combined_reward.float()
        batch_mean_reward = combined_reward_float.mean(dim=1, keepdim=True)
        # (batch, 1)
        combined_advantage = combined_reward_float - batch_mean_reward
        # (batch , 2*pomo)

        alpha= self.trainer_params.get('primal_transposed_weight', 0.5)

        primal_advantage = combined_advantage[:, :self.env.pomo_size]
        trans_advantage = combined_advantage[:, self.env.pomo_size:]

        # loss
        primal_loss = -primal_advantage * primal_log_prob
        trans_loss = -trans_advantage * trans_log_prob
        # (batch, pomo)
        primal_loss_mean = primal_loss.mean()
        trans_loss_mean = trans_loss.mean()
        final_loss_mean = alpha * primal_loss_mean + (1 - alpha) * trans_loss_mean

        # score
        max_pomo_reward, _ = combined_reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model.zero_grad()
        final_loss_mean.backward()
        self.optimizer.step()

        return score_mean.item(), final_loss_mean.item(), primal_loss_mean.item(), trans_loss_mean.item()
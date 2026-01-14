import torch
from logging import getLogger


from ACVRPEnv import ACVRPEnv as Env
from ACVRPModel import ACVRPModel as Model

from utils import *
class ACVRPTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.aug_transpose = self.tester_params['aug_transpose']
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(device)
        else:
            device = torch.device('cpu')
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(device)
        self.device = device

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        model_load = self.tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()

    def run(self):
        score = AverageMeter()
        aug_score = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            batch_score, batch_aug_score = self._test_one_batch(batch_size)
            score.update(batch_score, batch_size)
            aug_score.update(batch_aug_score, batch_size)
            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score.avg, aug_score.avg))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score.avg))

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        aug_enable = self.tester_params['augmentation_enable']
        aug_factor = self.tester_params['aug_factor'] if aug_enable else 1
        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_enable=aug_enable, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)


            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            # shape: (augmentation, batch, pomo)

            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)

            batch_reward = max_pomo_reward.max(dim=0)[0]  # get best results from augmentation
            # shape: (batch,)
            avg_reward_pre = -batch_reward.float().mean()  # negative sign to make positive value

            if self.aug_transpose:
                self.env.load_transposed_problems()
                t_reset_state, _, _ = self.env.reset()
                self.model.pre_forward(t_reset_state)

                t_state, t_reward, t_done = self.env.pre_step()
                while not t_done:
                    t_selected, _ = self.model(t_state)
                    t_state, t_reward, t_done = self.env.step(t_selected)

                t_aug_reward = t_reward.reshape(aug_factor, batch_size, self.env.pomo_size)
                # shape: (augmentation, batch, pomo)
                t_max_pomo_reward, _ = t_aug_reward.max(dim=2)
                # shape: (augmentation, batch)
                t_batch_reward, _ = t_max_pomo_reward.max(dim=0)
                # shape: (batch,)
                max_batch_reward = torch.max(batch_reward, t_batch_reward)
                avg_reward_pt = -max_batch_reward.float().mean()
            else:
                avg_reward_pt = avg_reward_pre
            return avg_reward_pre.item(), avg_reward_pt.item()


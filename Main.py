
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
from math import pi, sqrt, exp

from MuMIMOClass import envMuMIMO
from DQN import DQNAgent

if __name__ == "__main__":
    # Set device for PyTorch
    import torchA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Simulation Parameters
    EPISODES = 5000
    NumAntBS = 2
    NumEleIRS = 32
    NumUser = 1
    sigma2_BS = 0.1  # Noise level at BS side
    sigma2_UE = 0.5  # Noise level at UE side
    Pos_BS = np.array([0, 0, 10])  # Position of BS
    Pos_IRS = np.array([-2, 5, 5])  # Position of IRS
    MuMIMO_env = envMuMIMO(NumAntBS, NumEleIRS, NumUser)  # Environment
    batch_size = 8
    # State has two parts: [NumAntBS*NumUser*2 (real & imag), NumEleIRS*2 (real & imag)]
    state_size = [NumAntBS * NumUser * 2, NumEleIRS * 2]
    QuantLevel = 8  # Quantization level of phase shift

    ## Action Set
    ShiftCodebook = [
        np.exp(1j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS),
        np.exp(-1j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS),
        np.exp(3j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS),
        np.exp(-3j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS),
        np.exp(0j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS)
    ]
    ShiftCodebook = np.array(ShiftCodebook)
    action_size = ShiftCodebook.shape[0]

    ## Channel Dynamics
    block_duration = 1  # When block_duration>1, ESC will be applied
    BlockPerEpi = 3
    TimeTotal = BlockPerEpi * block_duration

    ## DQN Agent
    agent = DQNAgent(state_size, action_size, device)

    ## Initialization
    Rate_DQN_seq_episode = np.zeros(EPISODES)
    Rate_Random_seq_episode = np.zeros(EPISODES)

    # Initialize IRS reflection vector
    RefVector = np.exp(1j * pi * np.zeros((1, NumEleIRS)))
    RefVector_bench_random = RefVector.copy()
    Pilot = MuMIMO_env.DFT_matrix(NumUser)  # Pilot (DFT matrix)
    ArrayShape_BS = [NumAntBS, 1, 1]  # BS antenna array shape
    ArrayShape_IRS = [1, NumEleIRS, 1]  # IRS array shape
    ArrayShape_UE = [1, 1, 1]  # UE antenna shape

    Rate_Random_seq_block = np.zeros(BlockPerEpi)
    Rate_DQN_seq_block = np.zeros(BlockPerEpi)

    ###########################################
    for epi in range(EPISODES):
        # Randomize UE positions for each episode
        Pos_UE = np.array([[np.random.rand() * 10, np.random.rand() * 10, 1.5]], dtype=np.float64)


        # LoS components
        H_U2B_LoS, H_R2B_LoS, H_U2R_LoS = MuMIMO_env.H_GenFunLoS(
            Pos_BS, Pos_IRS, Pos_UE, ArrayShape_BS, ArrayShape_IRS, ArrayShape_UE
        )

        # For ESC performance tracking
        SumRate_seq = np.zeros(block_duration)

        for block in range(BlockPerEpi):
            # Generate NLoS components
            H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS = MuMIMO_env.H_GenFunNLoS(NumAntBS, NumEleIRS, NumUser)
            K = 10  # Rician K-factor
            H_U2B = sqrt(1/(K+1))*H_U2B_NLoS + sqrt(K/(K+1))*H_U2B_LoS
            H_R2B = sqrt(1/(K+1))*H_R2B_NLoS + sqrt(K/(K+1))*H_R2B_LoS
            H_U2R = sqrt(1/(K+1))*H_U2R_NLoS + sqrt(K/(K+1))*H_U2R_LoS

            # Aggregated channel
            H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])

            ### Benchmark: Random Reflection
            DFTcodebook = sqrt(NumEleIRS)*MuMIMO_env.DFT_matrix(NumEleIRS)
            random_index = random.randrange(len(DFTcodebook))
            RefVector_bench_random = DFTcodebook[random_index, :]
            H_synt_bench = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector_bench_random)
            Rate_bench, _, _ = MuMIMO_env.GetRewards(Pilot, H_synt_bench, sigma2_BS, sigma2_UE)
            random_rate = np.sum(Rate_bench)
            Rate_Random_seq_block[block] = random_rate

            ### Current State
            if block == 0:
                Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                H_est_vector = np.reshape(H_est, (1, NumAntBS * NumUser))
                # Concatenate real & imag -> shape becomes (1, 2*(NumAntBS*NumUser)) = (1, 8)
                stateA = np.concatenate((H_est_vector.real, H_est_vector.imag), axis=1)
                # IRS pattern real & imag -> (1, 2*NumEleIRS) = (1, 64)
                stateB = np.concatenate((RefVector.real, RefVector.imag), axis=1)
                Current_State = [stateA, stateB]
            else:
                Current_State = Next_State

            ### Action and Phase Adjustment
            Flag = 1  # Flag for ESC
            for i_time in range(block_duration):
                # Recompute rates
                Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                H_est_vector = np.reshape(H_est, (1, NumAntBS * NumUser))
                if i_time == 0:  # Coarse phase shift
                    if epi == 0:
                        action = random.randrange(len(ShiftCodebook))
                        act_type = 'random'
                    else:
                        action, act_type = agent.act(Current_State)

                    RefVector = RefVector * ShiftCodebook[action, :]  # Update phase shift
                    H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])

                    # Evaluate new Rate
                    Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                    Rate_est, _, _ = MuMIMO_env.GetRewards(Pilot, H_est, sigma2_BS, sigma2_UE)
                    SumRate_seq[i_time] = np.sum(Rate_est)

                else:  # Fine phase shift (dither)
                    if Flag == 1:
                        Dither = np.exp(1j * 2 * pi * 1 / (2 ** QuantLevel) *
                                        (np.random.randint(8, size=(1, NumEleIRS)) - 4))
                        RefVector = RefVector * Dither[0]
                        H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])
                        Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                        Rate_est, _, _ = MuMIMO_env.GetRewards(Pilot, H_est, sigma2_BS, sigma2_UE)
                        SumRate_seq[i_time] = np.sum(Rate_est)
                        if SumRate_seq[i_time] > SumRate_seq[i_time - 1]:
                            Flag = 1
                        else:
                            Flag = -1
                    else:
                        RefVector = RefVector * np.conjugate(Dither[0]) * np.conjugate(Dither[0])
                        H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])
                        Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                        Rate_est, _, _ = MuMIMO_env.GetRewards(Pilot, H_est, sigma2_BS, sigma2_UE)
                        SumRate_seq[i_time] = np.sum(Rate_est)
                        Flag = 1

            # Final evaluation for this block
            H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])
            Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
            Rate_DQN_seq_block[block] = np.sum(Rate)

            ### Reward
            if Rate_DQN_seq_block[block] > 10:
                Reward = Rate_DQN_seq_block[block]
            else:
                Reward = Rate_DQN_seq_block[block] - 100

            ### Next State
            H_est_vector = np.reshape(H_est, (1, NumAntBS * NumUser))
            nextStateA = np.concatenate((H_est_vector.real, H_est_vector.imag), axis=1)
            nextStateB = np.concatenate((RefVector.real, RefVector.imag), axis=1)
            Next_State = [nextStateA, nextStateB]

            ### Memorize and Train
            agent.memorize(Current_State, action, Reward, Next_State)
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)

        # Print moving averages every N episodes
        N = 64
        if epi >= N:
            print("episode: {}, e: {:.2f}, MovingAveRLearning:{:.4f}, MovingAveRandom:{:.4f}".format(
                epi, agent.epsilon,
                np.mean(Rate_DQN_seq_episode[epi-N:epi]),
                np.mean(Rate_Random_seq_episode[epi-N:epi])
            ))
        else:
            print("episode: {}, e: {:.2f}".format(epi, agent.epsilon))

        if epi % 20 == 0:
            agent.update_target_model()
            agent.save("IRS_DQN.pth")

        Rate_DQN_seq_episode[epi] = np.mean(Rate_DQN_seq_block)
        Rate_Random_seq_episode[epi] = np.mean(Rate_Random_seq_block)

    # Save the final model weights at the end of training
    agent.save("IRS_DQN_final.pth")

    # Save results to CSV and plot
    localtime = time.localtime(time.time())
    print(localtime)
    dataframe = pd.DataFrame({
        'Rate_DQN_seq_episode': Rate_DQN_seq_episode,
        'Rate_Random_seq_episode': Rate_Random_seq_episode
    })
    dataframe.to_csv("Rate" + str(localtime) + ".csv", index=False, sep=',')

    def get_moving_average(mylist, N):
        cumsum, moving_aves = [0], []
        for i, x in enumerate(mylist, 1):
            cumsum.append(cumsum[i - 1] + x)
            if i >= N:
                moving_ave = (cumsum[i] - cumsum[i - N]) / N
                moving_aves.append(moving_ave)
        return moving_aves

    plt.figure()
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Performance $P_m$ (bps/Hz)', fontsize=14)
    SumRate_seq_ave = get_moving_average(Rate_DQN_seq_episode, N)
    RandomRate_seq_ave = get_moving_average(Rate_Random_seq_episode, N)
    x = np.arange(len(SumRate_seq_ave)) + N
    plt.plot(x, SumRate_seq_ave, 'r-', linewidth=3)
    plt.plot(x, RandomRate_seq_ave, 'g-', linewidth=3)
    plt.legend(['ReLearning', 'Random'])
    plt.savefig('destination_path.eps', format='eps')
    plt.savefig('destination_path.pdf')
    plt.show()

from pathlib import Path
from gflownet.envs.verification_env import VerificationEnv
from gflownet.proxy.verification_proxy import VerificationProxy
import hydra
import torch
from plot import plot_function_coordinates_UMAP, plot_gfn_dimension_combinations, plot_function_segments

def main(plots=True):
    expert_states = ["slope[48,64] min[40,86] max[70,90] argmax[70,90] iqr_fast[86,127] median_fast[0,48] median_fast[86,127] mad[86,127]",
                     "mean[0,48] mean_change[48,64] mean_absolute_change[70,90] energy[86,127] extreme_points_count[40,90] mean[64,74] std[0,48] std[86,127]",]
    bootstrap_folder = Path("./csv_data/bootstrap")
    bootstrap_imgs_folder = bootstrap_folder / "images"
    bootstrap_imgs_folder.mkdir(parents=True, exist_ok=True)
    bootstrap_data_path = bootstrap_folder / "perturbed_expert_states.csv"
    noob_states = ["std[0,48] std[0,48] std[0,48] std[0,48] std[0,48] std[0,48] std[0,48] std[0,48]",
                    "mean[0,48] mean[0,48] mean[0,48] mean[0,48] mean[0,48] mean[0,48] mean[0,48] mean[0,48]",]

    env_config_path = Path("/home/dmd_user/Desktop/ECAA/gflownet/config/env/verification.yaml")
    with open(env_config_path, "r") as f:
        import yaml
        env_config = yaml.safe_load(f)


    env = VerificationEnv(data_path=env_config["data_path"], 
                          device="cpu", 
                          window_size=env_config["window_size"], 
                          resolution=env_config["resolution"],
                          min_function_width=env_config["min_function_width"], 
                          max_length=env_config["max_length"])

    perturbations = []

    for i, expert_state in enumerate(expert_states):
        expert_state = env.readable2state(expert_state)
        # add noise to intervals in expert_state to create perturbed states
        num_perturbations = 100

        perturbed_states = torch.zeros([num_perturbations, env.source.shape[0], env.source.shape[1]], dtype=torch.int16)

        for j in range(num_perturbations):
            perturbed_state = expert_state.clone()
            for action in perturbed_state:
                start = action[0].item()
                end = action[1].item()
                func = action[2].item()
                noise_start = torch.randint(-5, 6, (1,)).item()
                noise_end = torch.randint(-5, 6, (1,)).item()
                new_start = max(0, start + noise_start)
                new_end = min(env.resolution - 1, end + noise_end)
                # ensure min_function_width is maintained
                if new_end - new_start + 1 < env.min_function_width:
                    if noise_start < 0:
                        new_start = new_end - env.min_function_width + 1
                    else:
                        new_end = new_start + env.min_function_width - 1
                action[0] = new_start
                action[1] = new_end
                action[2] = func
            
            #shuffle the order of functions in perturbed_state
            perm = torch.randperm(perturbed_state.shape[0])
            perturbed_states[j] = perturbed_state[perm]
        perturbed_states = torch.cat([expert_state.unsqueeze(0), perturbed_states], dim=0)
        perturbations.append(perturbed_states)
    proxy_config_path = Path("/home/dmd_user/Desktop/ECAA/gflownet/config/proxy/verification.yaml")
    with open(proxy_config_path, "r") as f:
        import yaml
        proxy_config = yaml.safe_load(f)


    proxy = VerificationProxy(reward_min=1e-8, 
                              do_clip_rewards=False, 
                              device=proxy_config["device"], 
                              production_data=proxy_config["production_data"])

    proxy.setup(env)
    for s in noob_states:
        state = env.readable2state(s)
        r = proxy(state.unsqueeze(0))
        print(f"Noob state: {s} | Reward: {r.item()}")

    perturbed_expert_states = torch.cat(perturbations, dim=0)

    rewards = proxy(perturbed_expert_states)
    # if plots:
    #     print("Perturbed States and Rewards:")
    #     for i in range(perturbed_expert_states.shape[0]):
    #         state = perturbed_expert_states[i]
    #         readable = env.state2readable(state)
    #         reward = rewards[i].item()
    #         print(f"State {i+1}: {readable} | Reward: {reward}")

    #         plot_function_coordinates_UMAP(readable, reward, proxy, env, save_path=bootstrap_imgs_folder, fig_idx=i)
    #         plot_gfn_dimension_combinations(readable, reward, proxy, env, save_path=bootstrap_imgs_folder, fig_idx=i)
    #         plot_function_segments(readable, reward, proxy, env, save_path=bootstrap_imgs_folder, fig_idx=i)

    #save perturbed states to bootstrap_data_path as readable strings
    import pandas as pd
    df = pd.DataFrame({"idx": list(range(perturbed_expert_states.shape[0])),
        "readable": [env.state2readable(state) for state in perturbed_expert_states],
        "reward": [rewards[i].item() for i in range(perturbed_expert_states.shape[0])]
    })

    # df.to_csv(bootstrap_data_path, index=False)
    # print(f"Saved perturbed states to {bootstrap_data_path}")

    if plots:
        print(df)

if __name__ == "__main__":
    main()
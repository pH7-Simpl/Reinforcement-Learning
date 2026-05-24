import torch
import torch.optim as optim
import random
import math
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import gymnasium as gym
import numpy as np
import json
import os
from collections import namedtuple
from datetime import datetime

from q_network import DQN
from replay_memory import ReplayMemory
from IPython.display import clear_output

# Set up matplotlib untuk inline plotting
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Device setup
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def make_json_safe(obj):
    """Convert numpy/torch types ke Python native untuk JSON serializable."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj


def run_experiment(
    batch_size=128,
    gamma=0.99,
    eps_start=0.9,
    eps_end=0.01,
    eps_decay=2500,
    tau=0.005,
    lr=3e-4,
    memory_size=10000,
    num_episodes=None,
    hidden_size=128,
    plot_live=True,
    seed=None,
    device=device,
    save_dir='./experiment_results',
    save_model=True,
    experiment_name=None
):
    """
    SAMA PERSIS dengan baseline PyTorch DQN tutorial.
    Hanya dibungkus dalam fungsi agar hyperparameter bisa diubah.
    """

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"exp-{timestamp}-lr{lr}-decay{eps_decay}-bs{batch_size}"

    exp_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Save directory: {exp_dir}")
    print(f"{'='*60}")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions, hidden_size).to(device)
    target_net = DQN(n_observations, n_actions, hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayMemory(memory_size)

    steps_done = 0
    episode_durations = []

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)

        if show_result:
            plt.title(f'Result - {experiment_name}')
        else:
            plt.clf()
            plt.title(f'Training... - {experiment_name}')

        plt.xlabel('Episode')
        plt.ylabel('Duration')

        # Plot raw data dengan label
        plt.plot(durations_t.numpy(), color='green', alpha=0.6, label='Raw Duration')

        # Plot moving average dengan label
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color='red', linewidth=2, label='MA 100')

        # Threshold solved
        plt.axhline(y=475, color='blue', linestyle='--', alpha=0.5, label='Solved (475)')

        # Tampilkan legend
        plt.legend(loc='lower right')
        plt.ylim(0, 520)

        plt.pause(0.001)

        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    if num_episodes is None:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                if plot_live:
                    plot_durations()
                break

    env.close()

    if plot_live:
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()

    last_100_mean = sum(episode_durations[-100:]) / min(100, len(episode_durations))

    result = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'episode_durations': episode_durations,
        'hyperparameters': {
            'batch_size': batch_size,
            'gamma': gamma,
            'eps_start': eps_start,
            'eps_end': eps_end,
            'eps_decay': eps_decay,
            'tau': tau,
            'lr': lr,
            'plot_live': plot_live,
            'seed': seed,
            'num_episodes': num_episodes,
            'hidden_size': hidden_size,
            'memory_size': memory_size,
        },
        'stats': {
            'max_duration': max(episode_durations),
            'last_100_mean': last_100_mean,
            'episodes_to_475': next(
                (i for i, d in enumerate(episode_durations) if d >= 475), 
                None
            ),
            'total_episodes': len(episode_durations),
        }
    }

    # ========== AUTO-SAVE ==========

    # 1. JSON
    json_path = os.path.join(exp_dir, 'result.json')
    json_safe = {
        'experiment_name': result['experiment_name'],
        'timestamp': result['timestamp'],
        'episode_durations': [int(d) for d in result['episode_durations']],
        'hyperparameters': {k: make_json_safe(v) for k, v in result['hyperparameters'].items()},
        'stats': {k: make_json_safe(v) for k, v in result['stats'].items()}
    }

    with open(json_path, 'w') as f:
        json.dump(json_safe, f, indent=2)
    print(f"\n✓ Saved JSON: {json_path}")

    # 2. NPZ
    npz_path = os.path.join(exp_dir, 'result.npz')
    np.savez(npz_path,
             episode_durations=np.array(episode_durations),
             last_100_mean=last_100_mean,
             max_duration=max(episode_durations))
    print(f"✓ Saved NPZ: {npz_path}")

    # 3. Model
    if save_model:
        model_path = os.path.join(exp_dir, 'policy_net.pt')
        torch.save(policy_net.state_dict(), model_path)
        print(f"✓ Saved Model: {model_path}")

        config_path = os.path.join(exp_dir, 'model_config.json')
        model_config = {
            'n_observations': int(n_observations),
            'n_actions': int(n_actions),
            'hidden_size': int(hidden_size),
        }
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"✓ Saved Model Config: {config_path}")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episode_durations, alpha=0.3, color='green', label='Raw Duration')
    if len(episode_durations) >= 100:
        means = np.convolve(episode_durations, np.ones(100)/100, mode='valid')
        ax.plot(range(99, len(episode_durations)), means, color='red', linewidth=2, label='MA 100')
    ax.axhline(y=475, color='blue', linestyle='--', alpha=0.5, label='Solved (475)')
    ax.set_title(f"{experiment_name}\nLast100_mean={last_100_mean:.1f}")
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 520)
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, f"plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Plot: {plot_path}")

    print(f"\n{'='*60}")
    print(f"Experiment Complete: {experiment_name}")
    print(f"Last 100 mean: {last_100_mean:.1f}")
    print(f"Solved @ episode: {result['stats']['episodes_to_475']}")
    print(f"Max duration: {result['stats']['max_duration']}")
    print(f"{'='*60}")

    return result

def show_video(model, num_episodes=1, max_steps=500, seed=None, device='cpu'):
    """
    Jalankan simulasi CartPole dengan model trained.
    Tampilkan visualisasi live di notebook tiap step.

    Args:
        model: DQN model (sudah eval mode)
        num_episodes: Berapa episode ditampilkan
        max_steps: Max step per episode
        seed: Random seed (opsional)
        device: Device untuk tensor
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    if seed is not None:
        env.reset(seed=seed)

    model.eval()

    for episode in range(num_episodes):
        state, _ = env.reset()
        if seed is not None:
            env.reset(seed=seed + episode)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in range(max_steps):
            with torch.no_grad():
                action = model(state).max(1).indices.view(1, 1)

            observation, reward, terminated, truncated, _ = env.step(action.item())

            clear_output(wait=True)
            plt.imshow(env.render())
            plt.title(f"Episode {episode+1} | Step {t+1}")
            plt.axis('off')
            plt.show()

            if terminated or truncated:
                print(f"Episode {episode+1} selesai: {t+1} steps")
                break

            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    env.close()


def show_video_from_experiment(exp_dir, num_episodes=1, max_steps=500, seed=None, device='cpu'):
    """
    Load model dari folder eksperimen dan jalankan show_video().

    Args:
        exp_dir: Path folder eksperimen (yang ada policy_net.pt)
        num_episodes: Berapa episode ditampilkan
        max_steps: Max step per episode
        seed: Random seed (opsional)
        device: Device untuk tensor
    """
    from q_network import DQN

    # Load config
    config_path = os.path.join(exp_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Buat model dan load weights
    model = DQN(config['n_observations'], config['n_actions'], config.get('hidden_size', 128))
    model_path = os.path.join(exp_dir, 'policy_net.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"Loaded model from: {exp_dir}")
    print(f"Running {num_episodes} episode(s)...")

    show_video(model, num_episodes=num_episodes, max_steps=max_steps, seed=seed, device=device)


# ============================================
# FUNGSI LOAD & LIST
# ============================================

def load_experiment(exp_dir):
    json_path = os.path.join(exp_dir, 'result.json')
    with open(json_path, 'r') as f:
        result = json.load(f)

    model_path = os.path.join(exp_dir, 'policy_net.pt')
    config_path = os.path.join(exp_dir, 'model_config.json')

    if os.path.exists(model_path) and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = DQN(config['n_observations'], config['n_actions'], config.get('hidden_size', 128))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        result['model'] = model

    return result


def list_experiments(save_dir='./experiment_results'):
    if not os.path.exists(save_dir):
        return []

    experiments = []
    for name in sorted(os.listdir(save_dir)):
        exp_dir = os.path.join(save_dir, name)
        json_path = os.path.join(exp_dir, 'result.json')
        if os.path.isdir(exp_dir) and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            experiments.append({
                'name': name,
                'dir': exp_dir,
                'stats': data['stats'],
                'hyperparameters': data['hyperparameters'],
                'timestamp': data['timestamp']
            })

    return experiments


# ============================================
# FUNGSI PLOT BATCH COMPARISON
# ============================================

def plot_single_experiment(result, save_path=None, show=True):
    durations = result['episode_durations']
    params = result['hyperparameters']
    stats = result['stats']
    exp_name = result.get('experiment_name', 'Experiment')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(durations, alpha=0.3, color='green', label='Raw Duration')

    if len(durations) >= 100:
        means = np.convolve(durations, np.ones(100)/100, mode='valid')
        ax.plot(range(99, len(durations)), means, color='red', linewidth=2, label='MA 100')

    ax.axhline(y=475, color='blue', linestyle='--', alpha=0.7, label='Solved (475)')
    ax.set_title(f"{exp_name}\nlr={params['lr']}, decay={params['eps_decay']}, bs={params['batch_size']}\n"
                 f"Last100_mean={stats['last_100_mean']:.1f}, Solved@ep={stats['episodes_to_475']}")
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration (steps)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 520)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_experiments_comparison(results, save_path=None, show=True, max_cols=3):
    n = len(results)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n > 1 else [axes]

    for idx, (ax, result) in enumerate(zip(axes, results)):
        durations = result['episode_durations']
        params = result['hyperparameters']
        stats = result['stats']
        exp_name = result.get('experiment_name', f'Exp {idx+1}')

        ax.plot(durations, alpha=0.25, color='green', label='Raw')
        if len(durations) >= 100:
            means = np.convolve(durations, np.ones(100)/100, mode='valid')
            ax.plot(range(99, len(durations)), means, color='red', linewidth=2, label='MA 100')
        ax.axhline(y=475, color='blue', linestyle='--', alpha=0.5, label='Solved')
        ax.set_title(f"{exp_name}\n"
                     f"lr={params['lr']}, decay={params['eps_decay']}\n"
                     f"bs={params['batch_size']}\n"
                     f"mean={stats['last_100_mean']:.0f}, solve@ep={stats['episodes_to_475']}",
                     fontsize=9)
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel('Duration', fontsize=9)
        ax.set_ylim(0, 520)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='lower right')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('DQN Hyperparameter Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_ranking_bar(results, metric='last_100_mean', save_path=None, show=True):
    labels = []
    values = []

    for r in results:
        p = r['hyperparameters']
        label = f"lr={p['lr']}\ndecay={p['eps_decay']}\nbs={p['batch_size']}"
        labels.append(label)
        values.append(r['stats'][metric])

    sorted_pairs = sorted(zip(values, labels), reverse=True)
    values, labels = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    bars = ax.bar(range(len(values)), values, color=colors)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Experiment Ranking by {metric}')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig
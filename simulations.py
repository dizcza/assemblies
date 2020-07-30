# Default simulation library containing:
# - Basic projection simulations (convergence for different beta, etc)
# - Merge simulations (different betas)
# - Pattern completion simulations
# - Association simulations
# - simulations studying density in assemblies (higher than ambient p)

# Also contains methods for plotting saved results from some of these simulations
# (for figures).

import copy
import random
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import brain
import brain_util as bu
from overlap_sim import overlap_grand_sim, OVERLAP_RESULTS_PATH

RESULTS_DIR = Path(__file__).absolute().with_name("results")

PROJECT_RESULTS_PATH = RESULTS_DIR / 'project_results'
MERGE_BETAS_PATH = RESULTS_DIR / 'merge_betas'
ASSOCIATION_RESULTS_PATH = RESULTS_DIR / 'association_results'
PATTERN_COM_ITERATIONS_PATH = RESULTS_DIR / 'pattern_com_iterations'
DENSITY_RESULTS_PATH = RESULTS_DIR / 'density_results'


def project_sim(n=1000000, k=1000, p=0.01, beta=0.05, t=50):
    b = brain.Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(t - 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    return b.areas["A"].saved_w


def project_beta_sim(n=100000, k=317, p=0.01, t=100):
    results = {}
    for beta in [0.25, 0.1, 0.075, 0.05, 0.03, 0.01, 0.007, 0.005, 0.003,
                 0.001]:
        print("Working on " + str(beta) + "\n")
        out = project_sim(n, k, p, beta, t)
        results[beta] = out
    bu.sim_save(PROJECT_RESULTS_PATH, results)
    return results


def assembly_only_sim(n=100000, k=317, p=0.05, beta=0.05, project_iter=10):
    b = brain.Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter - 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    for i in range(5):
        b.project({}, {"A": ["A"]})
    return b.areas["A"].saved_w


# alpha = percentage of (random) final assembly neurons to try firing
def pattern_com(n=100000, k=317, p=0.05, beta=0.05, project_iter=10, alpha=0.5,
                comp_iter=1):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter - 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    # pick random subset of the neurons to fire
    subsample_size = int(k * alpha)
    subsample = random.sample(b.areas["A"].winners, subsample_size)
    b.areas["A"].winners = subsample
    for i in range(comp_iter):
        b.project({}, {"A": ["A"]})
    return b.areas["A"].saved_w, b.areas["A"].saved_winners


def pattern_com_repeated(n=100000, k=317, p=0.05, beta=0.05, project_iter=12,
                         alpha=0.4,
                         trials=3, max_recurrent_iter=10, resample=False):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter - 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})

    subsample_size = int(k * alpha)
    rounds_to_completion = []
    # pick random subset of the neurons to fire
    subsample = random.sample(b.areas["A"].winners, subsample_size)
    for trail in range(trials):
        if resample:
            subsample = random.sample(b.areas["A"].winners, subsample_size)
        b.areas["A"].winners = subsample
        rounds = 0
        while True:
            rounds += 1
            b.project({}, {"A": ["A"]})
            if (b.areas["A"].num_first_winners == 0) or (
                    rounds == max_recurrent_iter):
                break
        rounds_to_completion.append(rounds)
    saved_winners = b.areas["A"].saved_winners
    overlaps = bu.get_overlaps(saved_winners, project_iter - 1,
                               percentage=True)
    return overlaps, rounds_to_completion


def pattern_com_alphas(n=100000, k=317, p=0.01, beta=0.05,
                       alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               1.0], project_iter=25, comp_iter=5):
    b = brain.Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter - 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    results = {}
    A_winners = b.areas["A"].winners
    for alpha in alphas:
        # pick random subset of the neurons to fire
        subsample_size = int(k * alpha)
        b_copy = copy.deepcopy(b)
        subsample = random.sample(b_copy.areas["A"].winners, subsample_size)
        b_copy.areas["A"].winners = subsample
        for i in range(comp_iter):
            b_copy.project({}, {"A": ["A"]})
        final_winners = b_copy.areas["A"].winners
        o = bu.overlap(final_winners, A_winners)
        results[alpha] = float(o) / float(k)
    return results


def pattern_com_iterations(n=100000, k=317, p=0.01, beta=0.05, alpha=0.4,
                           comp_iter=8,
                           min_iter=20, max_iter=30):
    b = brain.Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(min_iter - 2):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    results = {}
    subsample_size = int(k * alpha)
    subsample = random.sample(b.areas["A"].winners, subsample_size)
    for i in range(min_iter, max_iter + 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        b_copy = copy.deepcopy(b)
        b_copy.areas["A"].winners = subsample
        for j in range(comp_iter):
            b_copy.project({}, {"A": ["A"]})
        o = bu.overlap(b_copy.areas["A"].winners, b.areas["A"].winners)
        results[i] = float(o) / float(k)
    bu.sim_save(PATTERN_COM_ITERATIONS_PATH, results)
    return results


# Sample command c_w,c_winners = bu.association_sim()
def associate(n=100000, k=317, p=0.05, beta=0.1, overlap_iter=10):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    b.add_stimulus("stimB", k)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    b.project({"stimA": ["A"], "stimB": ["B"]}, {})
    # Create assemblies A and B to stability
    for i in range(9):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A"], "B": ["B"]})
    b.project({"stimA": ["A"]}, {"A": ["A", "C"]})
    # Project A->C
    for i in range(9):
        b.project({"stimA": ["A"]},
                  {"A": ["A", "C"], "C": ["C"]})
    # Project B->C
    b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
    for i in range(9):
        b.project({"stimB": ["B"]},
                  {"B": ["B", "C"], "C": ["C"]})
    # Project both A,B to C
    b.project({"stimA": ["A"], "stimB": ["B"]},
              {"A": ["A", "C"], "B": ["B", "C"]})
    for i in range(overlap_iter - 1):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]})
    # Project just B
    b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
    for i in range(9):
        b.project({"stimB": ["B"]}, {"B": ["B", "C"], "C": ["C"]})
    return b


def association_sim(n=100000, k=317, p=0.05, beta=0.1, overlap_iter=10):
    b = associate(n, k, p, beta, overlap_iter)
    return b.areas["C"].saved_w, b.areas["C"].saved_winners


def association_grand_sim(n=100000, k=317, p=0.01, beta=0.05, min_iter=10,
                          max_iter=20):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    b.add_stimulus("stimB", k)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    b.project({"stimA": ["A"], "stimB": ["B"]}, {})
    # Create assemblies A and B to stability
    for i in range(9):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A"], "B": ["B"]})
    b.project({"stimA": ["A"]}, {"A": ["A", "C"]})
    # Project A->C
    for i in range(9):
        b.project({"stimA": ["A"]},
                  {"A": ["A", "C"], "C": ["C"]})
    # Project B->C
    b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
    for i in range(9):
        b.project({"stimB": ["B"]},
                  {"B": ["B", "C"], "C": ["C"]})
    # Project both A,B to C
    b.project({"stimA": ["A"], "stimB": ["B"]},
              {"A": ["A", "C"], "B": ["B", "C"]})
    for i in range(min_iter - 2):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]})
    results = {}
    for i in range(min_iter, max_iter + 1):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]})
        b_copy1 = copy.deepcopy(b)
        b_copy2 = copy.deepcopy(b)
        # in copy 1, project just A
        b_copy1.project({"stimA": ["A"]}, {})
        b_copy1.project({}, {"A": ["C"]})
        # in copy 2, project just B
        b_copy2.project({"stimB": ["B"]}, {})
        b_copy2.project({}, {"B": ["C"]})
        o = bu.overlap(b_copy1.areas["C"].winners, b_copy2.areas["C"].winners)
        results[i] = float(o) / float(k)
    bu.sim_save(ASSOCIATION_RESULTS_PATH, results)
    return results


def merge_sim(n=100_000, k=317, p=0.01, beta=0.05, max_t=50):
    b = brain.Brain(p)
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)

    b.project({"stimA": ["A"]}, {})
    b.project({"stimB": ["B"]}, {})
    b.project({"stimA": ["A"], "stimB": ["B"]},
              {"A": ["A", "C"], "B": ["B", "C"]})
    b.project({"stimA": ["A"], "stimB": ["B"]},
              {"A": ["A", "C"], "B": ["B", "C"], "C": ["C", "A", "B"]})
    for i in range(max_t - 1):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C", "A", "B"]})
    return b.areas["C"].saved_w


def merge_beta_sim(n=100000, k=317, p=0.01, t=100):
    results = {}
    for beta in [0.3, 0.2, 0.1, 0.075, 0.05]:
        print("Working on " + str(beta) + "\n")
        out = merge_sim(n, k, p, beta=beta, max_t=t)
        results[beta] = out
    bu.sim_save(MERGE_BETAS_PATH, results)
    return results


# UTILS FOR EVAL


def plot_project_sim(show=True, save="", show_legend=False,
                     use_text_font=True):
    if not PROJECT_RESULTS_PATH.exists():
        project_beta_sim()

    results = bu.sim_load(PROJECT_RESULTS_PATH)
    # fonts
    if use_text_font:
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    # 0.05 and 0.07 overlap almost exactly, pop 0.07
    results.pop(0.007)
    od = OrderedDict(sorted(results.items()))
    x = np.arange(100)
    print(x)
    for key, val in od.items():
        plt.plot(x, val, linewidth=0.7)
    if show_legend:
        plt.legend(list(od.keys()), loc='upper left')
    ax = plt.axes()
    ax.set_xticks([0, 10, 20, 50, 100])
    k = 317
    plt.yticks([k, 2 * k, 5 * k, 10 * k, 13 * k],
               ["k", "2k", "5k", "10k", "13k"])
    plt.xlabel(r'$t$')

    if not show_legend:
        for line, name in zip(ax.lines, list(od.keys())):
            y = line.get_ydata()[-1]
            ax.annotate(name, xy=(1, y), xytext=(6, 0), color=line.get_color(),
                        xycoords=ax.get_yaxis_transform(),
                        textcoords="offset points",
                        size=10, va="center")
    if show:
        plt.show()
    if not show and save != "":
        save = Path(save)
        save.parent.mkdir(exist_ok=True)
        plt.savefig(save)


def plot_merge_sim(show=True, save="", show_legend=False, use_text_font=True):
    if not MERGE_BETAS_PATH.exists():
        merge_beta_sim()

    results = bu.sim_load(MERGE_BETAS_PATH)
    # fonts
    if use_text_font:
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    od = OrderedDict(sorted(results.items()))
    x = np.arange(101)
    for key, val in od.items():
        plt.plot(x, val, linewidth=0.7)
    if show_legend:
        plt.legend(list(od.keys()), loc='upper left')
    ax = plt.axes()
    ax.set_xticks([0, 10, 20, 50, 100])
    k = 317
    plt.yticks([k, 2 * k, 5 * k, 10 * k, 13 * k],
               ["k", "2k", "5k", "10k", "13k"])
    plt.xlabel(r'$t$')

    if not show_legend:
        for line, name in zip(ax.lines, list(od.keys())):
            y = line.get_ydata()[-1]
            ax.annotate(name, xy=(1, y), xytext=(6, 0), color=line.get_color(),
                        xycoords=ax.get_yaxis_transform(),
                        textcoords="offset points",
                        size=10, va="center")
    if show:
        plt.show()
    if not show and save != "":
        save = Path(save)
        save.parent.mkdir(exist_ok=True)
        plt.savefig(save)


def plot_association(show=True, save="", use_text_font=True):
    if not ASSOCIATION_RESULTS_PATH.exists():
        association_grand_sim()

    results = bu.sim_load(ASSOCIATION_RESULTS_PATH)
    if use_text_font:
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    od = OrderedDict(sorted(results.items()))
    plt.plot(list(od.keys()), list(od.values()), linewidth=0.7)
    ax = plt.axes()
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["10%", "20%", "30%", "40%", "50%"])
    plt.xlabel(r'$t$')
    if show:
        plt.show()
    if not show and save != "":
        save = Path(save)
        save.parent.mkdir(exist_ok=True)
        plt.savefig(save)


def plot_pattern_com(show=True, save="", use_text_font=True):
    if not PATTERN_COM_ITERATIONS_PATH.exists():
        pattern_com_iterations()

    results = bu.sim_load(PATTERN_COM_ITERATIONS_PATH)
    if use_text_font:
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    od = OrderedDict(sorted(results.items()))
    plt.plot(list(od.keys()), list(od.values()), linewidth=0.7)
    ax = plt.axes()
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    plt.xlabel(r'$t$')
    if show:
        plt.show()
    if not show and save != "":
        save = Path(save)
        save.parent.mkdir(exist_ok=True)
        plt.savefig(save)


def plot_overlap(show=True, save="", use_text_font=True):
    if not OVERLAP_RESULTS_PATH.exists():
        overlap_grand_sim()

    results = bu.sim_load(OVERLAP_RESULTS_PATH)
    if use_text_font:
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    od = OrderedDict(sorted(results.items()))
    plt.plot(list(od.keys()), list(od.values()), linewidth=0.7)
    ax = plt.axes()
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8], ["", "20%", "40%", "60%", "80%"])
    plt.xlabel('overlap (assemblies)')
    plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
               ["", "5%", "10%", "15%", "20%", "25%", "30%"])
    plt.ylabel('overlap (projections)')
    if show:
        plt.show()
    if not show and save != "":
        save = Path(save)
        save.parent.mkdir(exist_ok=True)
        plt.savefig(save)


def density(n=100000, k=317, p=0.01, beta=0.05):
    b = brain.Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(9):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    conn = b.connectomes["A"]["A"]
    final_winners = b.areas["A"].winners
    edges = 0
    for i in final_winners:
        for j in final_winners:
            if conn[i][j] != 0:
                edges += 1
    return float(edges) / float(k ** 2)


def density_sim(n=100000, k=317, p=0.01,
                beta_values=[0, 0.025, 0.05, 0.075, 0.1]):
    results = {}
    for beta in beta_values:
        print("Working on " + str(beta) + "\n")
        out = density(n, k, p, beta)
        results[beta] = out
    bu.sim_save(DENSITY_RESULTS_PATH, results)
    return results


def plot_density_ee(show=True, save="", use_text_font=True):
    if not DENSITY_RESULTS_PATH.exists():
        density_sim()

    if use_text_font:
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
    od = bu.sim_load(DENSITY_RESULTS_PATH)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'assembly $p$')
    plt.plot(list(od.keys()), list(od.values()), linewidth=0.7)
    plt.plot(list(od.keys()), [0.01] * len(od), color='red',
             linestyle='dashed', linewidth=0.7)
    if show:
        plt.show()
    if not show and save != "":
        save = Path(save)
        save.parent.mkdir(exist_ok=True)
        plt.savefig(save)


if __name__ == '__main__':
    plot_project_sim(show=False, save='plots/project_sim.png')
    plot_merge_sim(show=False, save='plots/merge_sim.png')
    plot_association(show=False, save='plots/association.png')
    plot_pattern_com(show=False, save='plots/pattern_com.png')
    plot_overlap(show=False, save='plots/overlap.png')
    plot_density_ee(show=False, save='plots/density_ee.png')

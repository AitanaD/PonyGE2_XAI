from algorithm.parameters import params
from stats.stats import stats


def clean_stats():
    """
    Removes certain unnecessary stats from the stats.stats.stats dictionary
    to clean up the current run.
    
    :return: Nothing.
    """

    stats = {'gen': 0, 'total_inds': 0, 'regens': 0, 'invalids': 0, 'runtime_error': 0, 'unique_inds': 0, 'unused_search': 0, 'ave_genome_length': 0, 'max_genome_length': 0, 'min_genome_length': 0, 'ave_used_codons': 0, 'max_used_codons': 0, 'min_used_codons': 0, 'ave_tree_depth': 0, 'max_tree_depth': 0, 'min_tree_depth': 0, 'ave_tree_nodes': 0, 'max_tree_nodes': 0, 'min_tree_nodes': 0, 'ave_fitness': 0, 'best_fitness': 0, 'time_taken': 0, 'total_time': 0, 'time_adjust': 0}
    #trackers.best_ever = None
    """
    if not params['CACHE']:
        stats.pop('unique_inds')
        stats.pop('unused_search')

    if not params['MUTATE_DUPLICATES']:
        stats.pop('regens')
    """
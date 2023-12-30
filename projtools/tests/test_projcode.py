""" Projcode tests
"""

from projtools import projcode


assert starpower_dict_nan_count == 0

assert not any(top_actors['genders'].astype(str).eq('Unknown'))
assert not any(top_actors['race'].astype(str).eq('Unknown'))
assert not any(top_actors['birth_year'].astype(str).eq('Unknown'))

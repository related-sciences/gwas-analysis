from hypothesis import Phase

# Phases setting without shrinking for complex, conditional draws in
# which shrinking wastes time and adds little information
# (see https://hypothesis.readthedocs.io/en/latest/settings.html#hypothesis.settings.phases)
PHASES_NO_SHRINK = (Phase.explicit, Phase.reuse, Phase.generate, Phase.target)
from pysips.priors import load_corpus


def test_load_prior_corpus():
    corpus = load_corpus("wikipedia", max_samples=10)
    assert len(corpus) == 10

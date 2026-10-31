from cropml.data.dedup import cluster_near_duplicates, cluster_sizes, compute_phashes


def test_identical_images_cluster_together(tmp_path, make_image_file, make_sample):
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    make_image_file(p1, seed=1)
    make_image_file(p2, seed=1)  # same seed -> byte-identical content

    samples = [
        make_sample("ds/a.jpg", "tomato__healthy", path=p1),
        make_sample("ds/b.jpg", "tomato__healthy", path=p2),
    ]
    hashed = compute_phashes(samples)
    assignment = cluster_near_duplicates(hashed, max_distance=8)
    assert assignment["ds/a.jpg"] == assignment["ds/b.jpg"]


def test_different_images_do_not_cluster(tmp_path, make_image_file, make_sample):
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    make_image_file(p1, seed=1)
    make_image_file(p2, seed=99)

    samples = [
        make_sample("ds/a.jpg", "tomato__healthy", path=p1),
        make_sample("ds/b.jpg", "tomato__healthy", path=p2),
    ]
    hashed = compute_phashes(samples)
    assignment = cluster_near_duplicates(hashed, max_distance=8)
    assert assignment["ds/a.jpg"] != assignment["ds/b.jpg"]


def test_clustering_is_scoped_per_class(tmp_path, make_image_file, make_sample):
    """Identical image content in two different classes must NOT cluster -
    dedup only catches near-duplicates within the same class."""
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    make_image_file(p1, seed=1)
    make_image_file(p2, seed=1)

    samples = [
        make_sample("ds/a.jpg", "tomato__healthy", path=p1),
        make_sample("ds/b.jpg", "potato__healthy", path=p2),
    ]
    hashed = compute_phashes(samples)
    assignment = cluster_near_duplicates(hashed, max_distance=8)
    assert assignment["ds/a.jpg"] != assignment["ds/b.jpg"]


def test_unreadable_image_becomes_singleton_with_warning(
    tmp_path, monkeypatch, make_image_file, make_sample
):
    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_bytes(b"not a real image")
    good_path = tmp_path / "good.jpg"
    make_image_file(good_path, seed=1)

    samples = [
        make_sample("ds/corrupt.jpg", "tomato__healthy", path=bad_path),
        make_sample("ds/good.jpg", "tomato__healthy", path=good_path),
    ]
    import pytest

    with pytest.warns(UserWarning):
        hashed = compute_phashes(samples)
    assert hashed[0].phash is None
    assignment = cluster_near_duplicates(hashed, max_distance=8)
    assert assignment["ds/corrupt.jpg"] != assignment["ds/good.jpg"]


def test_cluster_sizes_counts_correctly(tmp_path, make_image_file, make_sample):
    paths = [tmp_path / f"{i}.jpg" for i in range(3)]
    make_image_file(paths[0], seed=1)
    make_image_file(paths[1], seed=1)  # dup of paths[0]
    make_image_file(paths[2], seed=99)  # unique

    samples = [
        make_sample(f"ds/{i}.jpg", "tomato__healthy", path=p) for i, p in enumerate(paths)
    ]
    hashed = compute_phashes(samples)
    assignment = cluster_near_duplicates(hashed, max_distance=8)
    sizes = cluster_sizes(assignment)
    assert sorted(sizes.values()) == [1, 2]
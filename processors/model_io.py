import os


def get_cached_features_file(args, target_file_path):
    target_file = target_file_path.split("/")[-1]
    input_dir = '/'.join(target_file_path.split("/")[:-1])
    # Load data features from cache or dataset file

    print("\n\n\ttarget is -", target_file, "\n")

    if not os.path.exists(os.path.join(input_dir, args.data_cache_dir_name)):
        os.makedirs(os.path.join(input_dir, args.data_cache_dir_name))

    cached_features_file_path = os.path.join(
        input_dir,
        args.data_cache_dir_name,
        "cached_{}_{}".format(
            target_file.split(".json")[0],
            args.data_cache_postfix
        )
    )

    return cached_features_file_path


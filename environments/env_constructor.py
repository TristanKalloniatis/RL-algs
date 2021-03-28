def make_envs(env_name: str, env_fn, env_kwargs):
    def _thunk():
        env = env_fn(env_name, **env_kwargs)
        return env

    return _thunk
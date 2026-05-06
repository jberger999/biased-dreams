import torch


@torch.no_grad()
def get_uncertainties(experiment,
                      states,
                      actions,
                      rollout_length):
    """
    Compute state-wise model uncertainty.
    """
    experiment._trainer._imagine_horizon = rollout_length
    uncertainties = experiment._trainer.compute_uncertainty(states,
                                                            actions,
                                                            align=False)    # state-action pairs are already aligned from get_priors
    return uncertainties.cpu()


@torch.no_grad()
def get_reconstructions(experiment,
                        states):
    """
    Decode the given states using every available decoder.
    """
    imagined_features = experiment._model.get_features(state=states)
    decoder_outs = [experiment._model.decoders[i](imagined_features)[0].cpu() for i in range(len(experiment._model.decoders))]
    return decoder_outs


@torch.no_grad()
def get_decoded_physical_states(experiment, states):
    """
    Decode the given states using physical state decoder only.
    """
    states_features = experiment._model.get_deterministic_features(state=states)
    return experiment._model.decoders[2](states_features)[0] # 0 = mean


@torch.no_grad()
def get_decoded_rewards(experiment, states):
    """
    Decode the given states using reward decoder only.
    """
    states_features = experiment._model.get_deterministic_features(state=states)
    return experiment._model.decoders[1](states_features)[0] # 0 = mean
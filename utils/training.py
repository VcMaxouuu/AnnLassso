import torch
import torch.nn as nn
import numpy as np
import utils


def apply_armijo_rule(X, y, model, parameters, criterion, intermediate_loss, lr, min_lr, lambda_, nu):
    with torch.no_grad():
        current_lr = lr
        success = False
        original_params = {name: (p.data.clone(), p.grad.data.clone()) for name, p in parameters}
        # --------------------
        # Armijo Rule for Learning Rate
        # --------------------
        while not success and current_lr >= min_lr:
            for name, p in parameters:
                p.data = utils.shrinkage_operator(original_params[name][0] - current_lr * original_params[name][1], lambda_ * current_lr, nu)
            output = model.forward(X)
            total_loss, _ = criterion(output, y, model.penalized_parameters())
            final_loss = total_loss.item()

            if final_loss <= intermediate_loss:
                success = True
            else:
                current_lr *= 0.5

        if not success:
            for name, p in parameters:
                p.data = original_params[name][0]



def training(model, X, y, lambda_, nu, rel_err, verbose, max_epochs=50000, type='reg'):
    min_lr = 1e-9
    lr_penalized = 0.001 / (len(model.hidden_dims)**2)

    unpenalized_parameters = [param for _, param in model.unpenalized_parameters()]
    optimizer = torch.optim.Adam(unpenalized_parameters, lr=0.0001)

    epoch = 0
    last_loss = np.inf

    criterion = utils.PenalizedLoss(lambda_, nu, type)

    while True:
        # --------------------
        # Gradient Descent for Unpenalized Parameters
        # --------------------
        model.zero_grad()
        output = model.forward(X)
        starting_total_loss, _ = criterion(output, y, model.penalized_parameters())
        starting_total_loss.backward()
        optimizer.step()
        # --------------------
        # Learning Rate Adaptation for Unpenalized Parameters
        # --------------------
        if starting_total_loss.item() > last_loss:
            for g in optimizer.param_groups:
                g['lr'] *= 0.5
            if verbose:
                print(f"\tEpoch {epoch} | Reduced unpenalized parameters learning rate")
        # --------------------
        # New Gradients Computation
        # --------------------
        model.zero_grad()
        output = model.forward(X)
        total_loss, bare_loss = criterion(output, y, model.penalized_parameters())
        intermediate_loss = total_loss.item()
        bare_loss.backward()
        # --------------------
        # ISTA for Penalized Parameters
        # --------------------
        W0, biases = [], []
        for name, p in model.penalized_parameters():
            if 'weight' in name:
                W0.append((name, p))
            else:
                biases.append((name, p))

        for parameters in [biases, W0]:
            apply_armijo_rule(X, y, model, parameters, criterion, intermediate_loss, lr_penalized, min_lr, lambda_, nu)
            model.zero_grad()
            output = model.forward(X)
            total_loss, bare_loss = criterion(output, y, model.penalized_parameters())
            bare_loss.backward()
            intermediate_loss = total_loss.item()

        # --------------------
        # Logging and Convergence Checks
        # --------------------
        if epoch % 5 == 0:
            if verbose and epoch % 50 == 0:
                print(f"\tEpoch {epoch} | Loss: {total_loss.item():.5f} | Bare loss: {bare_loss.item():.5f} | Important features: {model.important_features()}")

            loss_change = abs(total_loss.item() - last_loss) / abs(last_loss)
            if loss_change < rel_err:
                if verbose:
                    print(f"\n\tDescent stopped: Relative loss change below {rel_err}.")
                break

            last_loss = total_loss.item()

        epoch += 1


def sgd(model, X, y, lr, lambda_, nu, rel_err, verbose, max_epochs=20000, type='reg'):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    last_loss = np.inf
    epoch = 0
    criterion = utils.PenalizedLoss(lambda_, nu, type)

    model.train()
    while True:
        # -----------------
        # Gradient Descent
        # -----------------
        output = model.forward(X)
        total_loss, bare_loss = criterion(output, y, model.penalized_parameters())

        model.zero_grad()
        total_loss.backward()
        optimizer.step()
        # -----------------
        # Convergence Checks and Logging
        # -----------------
        if total_loss.item() == 0:
            if verbose: print(f"\n\tDescent stopped: Loss becomes zero.")
            break

        if verbose and epoch % 50 == 0:
                print(f"\tEpoch {epoch} | Loss: {total_loss.item():.5f} | Bare loss: {bare_loss.item():.5f}")

        if epoch % 5 == 0:
            loss_change = abs(total_loss.item() - last_loss) / abs(total_loss.item())
            if loss_change < rel_err:
                if verbose: print(f"\n\tDescent stopped: Relative loss change below {rel_err}.")
                break

            if total_loss.item() > last_loss:
                for group in optimizer.param_groups:
                    group['lr'] *= 0.5

            last_loss = total_loss.item()

        epoch += 1

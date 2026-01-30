import torch.nn as nn
import torch 

def train_model(model_conv, 
                train_dataloader, 
                validation_dataloader,
                BATCH_SIZE=32,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    learning_rate = 1e-3 * (BATCH_SIZE / 32)

    weight_decay = 1e-6
    patience = 10
    verbose_ct = 1
    num_epochs = 20  

    criterion = nn.CrossEntropyLoss()  # change if needed

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_conv.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    model_conv.to(device)

    # =====================
    # Early Stopping State
    # =====================
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    # =====================
    # Training Loop
    # =====================
    for epoch in tqdm(range(1, num_epochs + 1)):

        # ---- Train ----
        model_conv.train()
        train_loss = 0.0

        for x, y in train_dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model_conv(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_dataloader.dataset)

        # ---- Validation ----
        model_conv.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in validation_dataloader:
                x, y = x.to(device), y.to(device)
                preds = model_conv(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(validation_dataloader.dataset)

        # ---- Logging ----
        if verbose_ct and epoch % verbose_ct == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state_dict = {k: v.clone() for k, v in model_conv.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose_ct:
                print(f"Early stopping triggered at epoch {epoch}")
            break


    if best_state_dict is not None:
        model_conv.load_state_dict(best_state_dict)
def train_model(model, train_gen, val_gen):

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs= 20
    )

    model.save("models/best_model.keras")

    return history
# from    torch import optim
# import  numpy as np

# def loss_batch(model, loss_func, xb, yb, opt = None):
#     loss  = loss_func(model(xb), yb)

#     if opt is not None:
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#     return loss.item(), len(xb)

# def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
#     for epoch in range(epochs):
#         model.train()
#         for xb, yb in train_dl:
#             loss_batch(model, loss_func, xb, yb, opt)

#         model.eval()
#         with torch.no_grad():
#             losses, nums = zip(
#                 *[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl]
#             )

#         val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

#         print(f"Epoch {epoch+1}/{epochs} - Valid Loss: {val_loss:.4f}")

# model1      = Net()
# lr          = 0.1
# epochs      = 5
# loss_func   = F.cross_entropy
# opt         = optim.SGD(model1.parameters(), lr = lr, momentum = 0.9)

# fit(epochs, model1, loss_func, opt, train_dl, valid_dl)
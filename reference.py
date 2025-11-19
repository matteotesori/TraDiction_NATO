# #-------------------------------------------------------------------------------#
# #                                   MODEL                                       #
# #-------------------------------------------------------------------------------#

# import torch.nn as nn
# import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d( 1, 16, kernel_size = 3, stride = 1, padding = 1)
#         self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
#         self.conv3 = nn.Conv2d(16, 10, kernel_size = 3, stride = 1, padding = 1)

#     def forward(self, xb):
#         xb = xb.view(-1, 1, 28, 28)                                                     # Reshape input to (batch_size, 1, 28, 28)
#         xb = F.relu(self.conv1(xb))
#         xb = F.relu(self.conv2(xb))
#         xb = F.relu(self.conv3(xb))
#         xb = F.avg_pool2d(xb, 4)
#         return xb.view(-1, xb.size(1))                                                  # Flatten output to (batch_size, 10)
    
# #-------------------------------------------------------------------------------#
# #                                  TRAINING                                     #
# #-------------------------------------------------------------------------------#
# import  torch
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

# model1  = Net()
# lr      = 0.1
# epochs  = 5
# loss_func = F.cross_entropy
# opt     = optim.SGD(model1.parameters(), lr = lr, momentum = 0.9)

# fit(epochs, model1, loss_func, opt, train_dl, valid_dl)

# #-------------------------------------------------------------------------------#
# #                               CHECK MODEL                                     #
# #-------------------------------------------------------------------------------#

# x, y    = next(iter(valid_dl))
# y_pred  = model1(x)
# print(y_pred.argmax(dim = 1))
# print(y)

# #-------------------------------------------------------------------------------#
# #                             SAVE/LOAD MODEL                                   #
# #-------------------------------------------------------------------------------#

# PATH = "model.pth"
# torch.save(model1.state_dict(), PATH)

# model2 = Net()
# model2.load_state_dict(torch.load(PATH))

# y_pred = model2(x)
# print(y_pred.argmax(dim = 1))
# print(y)
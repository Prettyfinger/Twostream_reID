from skimage import io
img=io.imread('/home/mcii216/fmx/RGB.jpg')
io.imshow(img)

# Increment = -20 / 100
# rgbMax = max(R(i, j), max(G(i, j), B(i, j)))
# rgbMin = min(R(i, j), min(G(i, j), B(i, j)))
# Delta = (rgbMax - rgbMin) / 255
# value = (rgbMax + rgbMin) / 255
# L = value / 2
# if (L < 0.5):
#     S = Delta / value
# else:
#     S = Delta / (2 - value)
#
#
# if (Increment >= 0) :
#     if ((Increment + S) >= 1):
#         alpha = S
#     else:
#         alpha = 1 - Increment
#
#     alpha = 1 / alpha - 1
#     R_new(i, j) = R(i, j) + (R(i, j) - L * 255) * alpha
#     G_new(i, j) = G(i, j) + (G(i, j) - L * 255) * alpha
#     B_new(i, j) = B(i, j) + (B(i, j) - L * 255) * alpha
# else :
#     alpha = Increment
#     R_new(i, j) = L * 255 + (R(i, j) - L * 255) * (1 + alpha)
#     G_new(i, j) = L * 255 + (G(i, j) - L * 255) * (1 + alpha)
#     B_new(i, j) = L * 255 + (B(i, j) - L * 255) * (1 + alpha)

io.imsave('/home/mcii216/fmx/RGB2.png',img)

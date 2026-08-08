import qrcode

# 生成二维码
data = "156+123+516+231"
img = qrcode.make(data)

# 保存图片
img.save("my_qrcode.png")
print("二维码已保存为 my_qrcode.png")
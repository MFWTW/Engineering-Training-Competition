import qrcode

# 生成二维码
data = "246+132+642+231"
img = qrcode.make(data)

# 保存图片
img.save("my_qrcode4.png")
print("二维码已保存为 my_qrcode.png")
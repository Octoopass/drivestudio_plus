"""Package ``drive2sim.engine``: điều phối việc chạy một approach (ARCHITECTURE mục 4.5).

Gồm ``stage`` (khuôn mẫu của stage), ``registry`` (đăng ký stage), ``resolve`` (quyết định danh
sách stage từ approach và lựa chọn), ``workdir`` (đường dẫn trong thư mục làm việc), ``cache``
(cache kết quả dùng chung) và ``run`` (tạo và chạy run). CLI và Studio gọi engine; engine không
biết bên trong từng stage.
"""

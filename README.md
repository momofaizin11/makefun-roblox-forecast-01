# roblox-up-and-coming-forecast-01

Prediksi 10 besar chart Roblox "Up and Coming" (US/computer) untuk 72 jam setelah deadline: output `forecast.json` berisi `probabilities` (`universe_id` × `place` × `p`) plus `top10`.
Jalankan lokal: `MAKEFUN_INPUT_DIR=../roblox-up-and-coming-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py`
Uji kontrak output tanpa file input (histori sintetis, deterministik): `python3 main.py --selftest`
Model: kernel difusi peringkat di sekitar posisi terakhir tiap game, dikali faktor pemain (growth, ukuran, umur, kehadiran di chart lain), dinormalkan per tempat lalu dikali decay horizon — stdlib saja, deterministik, <1 detik.
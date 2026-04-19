module {
  aie.device(npu2) {
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %out_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "out_7_cons_prod_lock_0"}
    %out_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "out_7_cons_cons_lock_0"}
    %out_7_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_7_buff_0"} : memref<256xbf16> 
    %out_7_buff_1 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_7_buff_1"} : memref<256xbf16> 
    %out_7_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 2 : i32, sym_name = "out_7_prod_lock_0"}
    %out_7_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "out_7_cons_lock_0"}
    %out_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 6) {init = 0 : i32, sym_name = "out_6_cons_prod_lock_0"}
    %out_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 7) {init = 0 : i32, sym_name = "out_6_cons_cons_lock_0"}
    %out_6_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_6_buff_0"} : memref<256xbf16> 
    %out_6_buff_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_6_buff_1"} : memref<256xbf16> 
    %out_6_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 2 : i32, sym_name = "out_6_prod_lock_0"}
    %out_6_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "out_6_cons_lock_0"}
    %out_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "out_5_cons_prod_lock_0"}
    %out_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "out_5_cons_cons_lock_0"}
    %out_5_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_5_buff_0"} : memref<256xbf16> 
    %out_5_buff_1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_5_buff_1"} : memref<256xbf16> 
    %out_5_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 2 : i32, sym_name = "out_5_prod_lock_0"}
    %out_5_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "out_5_cons_lock_0"}
    %out_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "out_4_cons_prod_lock_0"}
    %out_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "out_4_cons_cons_lock_0"}
    %out_4_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_4_buff_0"} : memref<256xbf16> 
    %out_4_buff_1 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_4_buff_1"} : memref<256xbf16> 
    %out_4_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 2 : i32, sym_name = "out_4_prod_lock_0"}
    %out_4_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "out_4_cons_lock_0"}
    %out_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 6) {init = 0 : i32, sym_name = "out_3_cons_prod_lock_0"}
    %out_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 7) {init = 0 : i32, sym_name = "out_3_cons_cons_lock_0"}
    %out_3_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_3_buff_0"} : memref<256xbf16> 
    %out_3_buff_1 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_3_buff_1"} : memref<256xbf16> 
    %out_3_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 2 : i32, sym_name = "out_3_prod_lock_0"}
    %out_3_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "out_3_cons_lock_0"}
    %out_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 0 : i32, sym_name = "out_2_cons_prod_lock_0"}
    %out_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "out_2_cons_cons_lock_0"}
    %out_2_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_2_buff_0"} : memref<256xbf16> 
    %out_2_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_2_buff_1"} : memref<256xbf16> 
    %out_2_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "out_2_prod_lock_0"}
    %out_2_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "out_2_cons_lock_0"}
    %out_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "out_1_cons_prod_lock_0"}
    %out_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "out_1_cons_cons_lock_0"}
    %out_1_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_1_buff_0"} : memref<256xbf16> 
    %out_1_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_1_buff_1"} : memref<256xbf16> 
    %out_1_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "out_1_prod_lock_0"}
    %out_1_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "out_1_cons_lock_0"}
    %out_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "out_0_cons_prod_lock_0"}
    %out_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "out_0_cons_cons_lock_0"}
    %out_0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_0_buff_0"} : memref<256xbf16> 
    %out_0_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_0_buff_1"} : memref<256xbf16> 
    %out_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "out_0_prod_lock_0"}
    %out_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "out_0_cons_lock_0"}
    %in2_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_7_cons_buff_0"} : memref<256xbf16> 
    %in2_7_cons_buff_1 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_7_cons_buff_1"} : memref<256xbf16> 
    %in2_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "in2_7_cons_prod_lock_0"}
    %in2_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "in2_7_cons_cons_lock_0"}
    %in2_7_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "in2_7_prod_lock_0"}
    %in2_7_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "in2_7_cons_lock_0"}
    %in2_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_6_cons_buff_0"} : memref<256xbf16> 
    %in2_6_cons_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_6_cons_buff_1"} : memref<256xbf16> 
    %in2_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "in2_6_cons_prod_lock_0"}
    %in2_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "in2_6_cons_cons_lock_0"}
    %in2_6_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "in2_6_prod_lock_0"}
    %in2_6_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "in2_6_cons_lock_0"}
    %in2_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_5_cons_buff_0"} : memref<256xbf16> 
    %in2_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_5_cons_buff_1"} : memref<256xbf16> 
    %in2_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "in2_5_cons_prod_lock_0"}
    %in2_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "in2_5_cons_cons_lock_0"}
    %in2_5_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "in2_5_prod_lock_0"}
    %in2_5_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "in2_5_cons_lock_0"}
    %in2_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_4_cons_buff_0"} : memref<256xbf16> 
    %in2_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_4_cons_buff_1"} : memref<256xbf16> 
    %in2_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "in2_4_cons_prod_lock_0"}
    %in2_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "in2_4_cons_cons_lock_0"}
    %in2_4_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "in2_4_prod_lock_0"}
    %in2_4_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "in2_4_cons_lock_0"}
    %in2_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_3_cons_buff_0"} : memref<256xbf16> 
    %in2_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_3_cons_buff_1"} : memref<256xbf16> 
    %in2_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "in2_3_cons_prod_lock_0"}
    %in2_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "in2_3_cons_cons_lock_0"}
    %in2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "in2_3_prod_lock_0"}
    %in2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "in2_3_cons_lock_0"}
    %in2_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_2_cons_buff_0"} : memref<256xbf16> 
    %in2_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_2_cons_buff_1"} : memref<256xbf16> 
    %in2_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "in2_2_cons_prod_lock_0"}
    %in2_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "in2_2_cons_cons_lock_0"}
    %in2_2_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "in2_2_prod_lock_0"}
    %in2_2_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "in2_2_cons_lock_0"}
    %in2_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_1_cons_buff_0"} : memref<256xbf16> 
    %in2_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_1_cons_buff_1"} : memref<256xbf16> 
    %in2_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "in2_1_cons_prod_lock_0"}
    %in2_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "in2_1_cons_cons_lock_0"}
    %in2_1_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "in2_1_prod_lock_0"}
    %in2_1_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "in2_1_cons_lock_0"}
    %in2_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_0_cons_buff_0"} : memref<256xbf16> 
    %in2_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_0_cons_buff_1"} : memref<256xbf16> 
    %in2_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "in2_0_cons_prod_lock_0"}
    %in2_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "in2_0_cons_cons_lock_0"}
    %in2_0_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "in2_0_prod_lock_0"}
    %in2_0_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "in2_0_cons_lock_0"}
    %in1_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_7_cons_buff_0"} : memref<256xbf16> 
    %in1_7_cons_buff_1 = aie.buffer(%tile_1_5) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_7_cons_buff_1"} : memref<256xbf16> 
    %in1_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "in1_7_cons_prod_lock_0"}
    %in1_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "in1_7_cons_cons_lock_0"}
    %in1_7_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "in1_7_prod_lock_0"}
    %in1_7_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "in1_7_cons_lock_0"}
    %in1_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_6_cons_buff_0"} : memref<256xbf16> 
    %in1_6_cons_buff_1 = aie.buffer(%tile_1_4) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_6_cons_buff_1"} : memref<256xbf16> 
    %in1_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "in1_6_cons_prod_lock_0"}
    %in1_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "in1_6_cons_cons_lock_0"}
    %in1_6_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "in1_6_prod_lock_0"}
    %in1_6_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "in1_6_cons_lock_0"}
    %in1_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_5_cons_buff_0"} : memref<256xbf16> 
    %in1_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_5_cons_buff_1"} : memref<256xbf16> 
    %in1_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "in1_5_cons_prod_lock_0"}
    %in1_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "in1_5_cons_cons_lock_0"}
    %in1_5_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "in1_5_prod_lock_0"}
    %in1_5_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "in1_5_cons_lock_0"}
    %in1_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_4_cons_buff_0"} : memref<256xbf16> 
    %in1_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_4_cons_buff_1"} : memref<256xbf16> 
    %in1_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "in1_4_cons_prod_lock_0"}
    %in1_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "in1_4_cons_cons_lock_0"}
    %in1_4_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "in1_4_prod_lock_0"}
    %in1_4_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "in1_4_cons_lock_0"}
    %in1_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_3_cons_buff_0"} : memref<256xbf16> 
    %in1_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_3_cons_buff_1"} : memref<256xbf16> 
    %in1_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "in1_3_cons_prod_lock_0"}
    %in1_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "in1_3_cons_cons_lock_0"}
    %in1_3_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "in1_3_prod_lock_0"}
    %in1_3_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "in1_3_cons_lock_0"}
    %in1_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_2_cons_buff_0"} : memref<256xbf16> 
    %in1_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_2_cons_buff_1"} : memref<256xbf16> 
    %in1_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "in1_2_cons_prod_lock_0"}
    %in1_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "in1_2_cons_cons_lock_0"}
    %in1_2_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "in1_2_prod_lock_0"}
    %in1_2_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "in1_2_cons_lock_0"}
    %in1_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_1_cons_buff_0"} : memref<256xbf16> 
    %in1_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_1_cons_buff_1"} : memref<256xbf16> 
    %in1_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "in1_1_cons_prod_lock_0"}
    %in1_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in1_1_cons_cons_lock_0"}
    %in1_1_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "in1_1_prod_lock_0"}
    %in1_1_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "in1_1_cons_lock_0"}
    %in1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_0_cons_buff_0"} : memref<256xbf16> 
    %in1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_0_cons_buff_1"} : memref<256xbf16> 
    %in1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in1_0_cons_prod_lock_0"}
    %in1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in1_0_cons_cons_lock_0"}
    %in1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "in1_0_prod_lock_0"}
    %in1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in1_0_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_1_5, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %tile_0_5, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %tile_1_3, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_1_4, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %tile_1_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %shim_noc_tile_0_0, DMA : 1)
    aie.flow(%tile_0_5, DMA : 0, %shim_noc_tile_3_0, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_1_3, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_1_4, DMA : 0, %shim_noc_tile_2_0, DMA : 1)
    aie.flow(%tile_1_5, DMA : 0, %shim_noc_tile_1_0, DMA : 1)
    func.func private @op13_eltwise_add_bf16_vector(memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) attributes {link_with = "op13_add.o"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 33280 : i32, mem_bank = 2 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c256_i32 = arith.constant 256 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i32
      cf.switch %4 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%in1_0_cons_buff_0 : memref<256xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%in1_0_cons_buff_1 : memref<256xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%in1_0_cons_buff_0 : memref<256xbf16>)
    ^bb6(%5: memref<256xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%in2_0_cons_buff_0 : memref<256xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%in2_0_cons_buff_1 : memref<256xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%in2_0_cons_buff_0 : memref<256xbf16>)
    ^bb10(%9: memref<256xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%out_0_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%out_0_buff_0 : memref<256xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%out_0_buff_1 : memref<256xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%out_0_buff_0 : memref<256xbf16>)
    ^bb14(%13: memref<256xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op13_eltwise_add_bf16_vector(%5, %9, %13, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%out_0_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous0[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["op13_add.o"]}
    %_anonymous1 = aie.buffer(%tile_0_3) {address = 33280 : i32, mem_bank = 2 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c256_i32 = arith.constant 256 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous1[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i32
      cf.switch %4 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%in1_1_cons_buff_0 : memref<256xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%in1_1_cons_buff_1 : memref<256xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%in1_1_cons_buff_0 : memref<256xbf16>)
    ^bb6(%5: memref<256xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%in2_1_cons_buff_0 : memref<256xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%in2_1_cons_buff_1 : memref<256xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%in2_1_cons_buff_0 : memref<256xbf16>)
    ^bb10(%9: memref<256xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%out_1_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%out_1_buff_0 : memref<256xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%out_1_buff_1 : memref<256xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%out_1_buff_0 : memref<256xbf16>)
    ^bb14(%13: memref<256xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op13_eltwise_add_bf16_vector(%5, %9, %13, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%out_1_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous1[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["op13_add.o"]}
    %_anonymous2 = aie.buffer(%tile_0_4) {address = 33280 : i32, mem_bank = 2 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c256_i32 = arith.constant 256 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous2[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i32
      cf.switch %4 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%in1_2_cons_buff_0 : memref<256xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%in1_2_cons_buff_1 : memref<256xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%in1_2_cons_buff_0 : memref<256xbf16>)
    ^bb6(%5: memref<256xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%in2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%in2_2_cons_buff_0 : memref<256xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%in2_2_cons_buff_1 : memref<256xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%in2_2_cons_buff_0 : memref<256xbf16>)
    ^bb10(%9: memref<256xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%out_2_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%out_2_buff_0 : memref<256xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%out_2_buff_1 : memref<256xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%out_2_buff_0 : memref<256xbf16>)
    ^bb14(%13: memref<256xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op13_eltwise_add_bf16_vector(%5, %9, %13, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_2_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%in2_2_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%out_2_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous2[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["op13_add.o"]}
    %_anonymous3 = aie.buffer(%tile_0_5) {address = 33280 : i32, mem_bank = 2 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c256_i32 = arith.constant 256 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous3[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i32
      cf.switch %4 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%in1_3_cons_buff_0 : memref<256xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%in1_3_cons_buff_1 : memref<256xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%in1_3_cons_buff_0 : memref<256xbf16>)
    ^bb6(%5: memref<256xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%in2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%in2_3_cons_buff_0 : memref<256xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%in2_3_cons_buff_1 : memref<256xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%in2_3_cons_buff_0 : memref<256xbf16>)
    ^bb10(%9: memref<256xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%out_3_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%out_3_buff_0 : memref<256xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%out_3_buff_1 : memref<256xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%out_3_buff_0 : memref<256xbf16>)
    ^bb14(%13: memref<256xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op13_eltwise_add_bf16_vector(%5, %9, %13, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_3_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%in2_3_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%out_3_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous3[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["op13_add.o"]}
    %_anonymous4 = aie.buffer(%tile_1_2) {address = 33280 : i32, mem_bank = 2 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c256_i32 = arith.constant 256 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous4[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i32
      cf.switch %4 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%in1_4_cons_buff_0 : memref<256xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%in1_4_cons_buff_1 : memref<256xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%in1_4_cons_buff_0 : memref<256xbf16>)
    ^bb6(%5: memref<256xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%in2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%in2_4_cons_buff_0 : memref<256xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%in2_4_cons_buff_1 : memref<256xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%in2_4_cons_buff_0 : memref<256xbf16>)
    ^bb10(%9: memref<256xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%out_4_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%out_4_buff_0 : memref<256xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%out_4_buff_1 : memref<256xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%out_4_buff_0 : memref<256xbf16>)
    ^bb14(%13: memref<256xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op13_eltwise_add_bf16_vector(%5, %9, %13, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_4_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%in2_4_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous4[%c1] : memref<3xi32>
      aie.use_lock(%out_4_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous4[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["op13_add.o"]}
    %_anonymous5 = aie.buffer(%tile_1_3) {address = 33280 : i32, mem_bank = 2 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_1_3 = aie.core(%tile_1_3) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c256_i32 = arith.constant 256 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous5[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i32
      cf.switch %4 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%in1_5_cons_buff_0 : memref<256xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%in1_5_cons_buff_1 : memref<256xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%in1_5_cons_buff_0 : memref<256xbf16>)
    ^bb6(%5: memref<256xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%in2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%in2_5_cons_buff_0 : memref<256xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%in2_5_cons_buff_1 : memref<256xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%in2_5_cons_buff_0 : memref<256xbf16>)
    ^bb10(%9: memref<256xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%out_5_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%out_5_buff_0 : memref<256xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%out_5_buff_1 : memref<256xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%out_5_buff_0 : memref<256xbf16>)
    ^bb14(%13: memref<256xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op13_eltwise_add_bf16_vector(%5, %9, %13, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_5_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%in2_5_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%out_5_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous5[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["op13_add.o"]}
    %_anonymous6 = aie.buffer(%tile_1_4) {address = 33280 : i32, mem_bank = 2 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c256_i32 = arith.constant 256 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous6[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i32
      cf.switch %4 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%in1_6_cons_buff_0 : memref<256xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%in1_6_cons_buff_1 : memref<256xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%in1_6_cons_buff_0 : memref<256xbf16>)
    ^bb6(%5: memref<256xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%in2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%in2_6_cons_buff_0 : memref<256xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%in2_6_cons_buff_1 : memref<256xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%in2_6_cons_buff_0 : memref<256xbf16>)
    ^bb10(%9: memref<256xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%out_6_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%out_6_buff_0 : memref<256xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%out_6_buff_1 : memref<256xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%out_6_buff_0 : memref<256xbf16>)
    ^bb14(%13: memref<256xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op13_eltwise_add_bf16_vector(%5, %9, %13, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_6_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%in2_6_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%out_6_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous6[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["op13_add.o"]}
    %_anonymous7 = aie.buffer(%tile_1_5) {address = 33280 : i32, mem_bank = 2 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_1_5 = aie.core(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c256_i32 = arith.constant 256 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous7[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i32
      cf.switch %4 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%in1_7_cons_buff_0 : memref<256xbf16>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%in1_7_cons_buff_1 : memref<256xbf16>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%in1_7_cons_buff_0 : memref<256xbf16>)
    ^bb6(%5: memref<256xbf16>):  // 3 preds: ^bb3, ^bb4, ^bb5
      aie.use_lock(%in2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%in2_7_cons_buff_0 : memref<256xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%in2_7_cons_buff_1 : memref<256xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%in2_7_cons_buff_0 : memref<256xbf16>)
    ^bb10(%9: memref<256xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%out_7_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%out_7_buff_0 : memref<256xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%out_7_buff_1 : memref<256xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%out_7_buff_0 : memref<256xbf16>)
    ^bb14(%13: memref<256xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op13_eltwise_add_bf16_vector(%5, %9, %13, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_7_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%in2_7_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%out_7_cons_lock_0, Release, 1)
      %24 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c2_i32 : i32
      %27 = arith.subi %25, %c2_i32 : i32
      %28 = arith.select %26, %27, %25 : i32
      memref.store %28, %_anonymous7[%c2] : memref<3xi32>
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_files = ["op13_add.o"]}
    aie.runtime_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @in1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @in1_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @in2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @in1_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @in2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @in1_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @in2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @in1_4_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @in2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @in1_5_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @in2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @in1_6_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @in2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @in1_7_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @in2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @out_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @out_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @out_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @out_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @out_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @out_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @out_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @out_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
    }
    aie.shim_dma_allocation @in1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_1_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_1_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_1_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_1_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_1_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_2_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_2_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_2_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_2_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_2_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_2_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_2_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_3_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_3_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_3_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_3_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_3_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_3_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_3_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_4_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_4_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_4_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_4_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_4_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_4_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_4_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_5_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_5_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_5_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_5_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_5_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_5_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_5_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_6_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_6_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_6_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_6_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_6_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_6_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_6_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_7_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_7_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_7_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_7_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_7_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_7_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_7_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in2_0_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @in2_1_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @in2_2_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @in2_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @in2_4_shim_alloc(%shim_noc_tile_7_0, MM2S, 1)
    aie.shim_dma_allocation @in2_5_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @in2_6_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @in2_7_shim_alloc(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @out_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out_1_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @out_2_shim_alloc(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @out_3_shim_alloc(%shim_noc_tile_3_0, S2MM, 1)
    aie.shim_dma_allocation @out_4_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @out_5_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @out_6_shim_alloc(%shim_noc_tile_2_0, S2MM, 1)
    aie.shim_dma_allocation @out_7_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}

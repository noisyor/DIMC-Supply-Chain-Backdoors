module simulated_DIMC_tb();

    // Parameters
    parameter ADDR_WIDTH = 5;
    parameter NUM_ROW = 1 << ADDR_WIDTH;
    parameter NUM_COL = 32;
    parameter OUT_WIDTH = NUM_COL * 4;
    parameter NUM_WEIGHT_CONFIG = 2;

    localparam WEIGHT_CONFIG_WIDTH = $clog2(NUM_WEIGHT_CONFIG);
    // Inputs
    logic clk, reset, write_enable, mac_enable, msb_iact, finish_iact_n;
    logic [ADDR_WIDTH-1:0] addr;
    logic [NUM_COL-1:0] weight_in;
    logic [NUM_ROW-1:0] iact;
    logic [WEIGHT_CONFIG_WIDTH-1:0] weight_config;

    // Outputs
    logic [OUT_WIDTH-1:0] oact;

    // Instantiate the simulated_DIMC module
    dimc_bank_row_acc_32x32_wReg #(
    ) dut (
        .clk(clk),
        // .reset(reset),
        .write_addr_in(addr),
        .write_data_in(weight_in),
        .activation_in(iact),
        .write_en_in(write_enable),
        .weight_config_in(weight_config),
        .MSB_in(msb_iact),
        .finish_n_in(finish_iact_n),
        .Output_out(oact)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Monitor for printing oact
initial begin
    $monitor("Time\t%t data_out\t0h%h", $time, oact);
end

    // Testbench code
    initial begin
        // Initialize inputs
        $dumpfile("DIMC_kernel_waveform.vcd");
        $dumpvars(0, simulated_DIMC_tb);
        clk = 1;
        write_enable = 0;
        addr = 0;
        weight_in = 0;
        iact = 32'h00000000;
        weight_config = 1'b0;
        msb_iact = 0;
        finish_iact_n = 1;
        #20;

        write_enable = 1;
        #10;
        write_enable = 0;
        // Initialize array with all 1s
        // for (int i = 0; i < NUM_ROW; i++) begin
        //     for (int j = 0; j < NUM_COL; j++) begin
        //         if (j % 8 == 0)
        //             dut.array[i][j] = 1;
        //         else
        //         dut.array[i][j] = 0;
        //     end
        // end
        $readmemb("../../data_weight.txt", dut.array);

        // Wait for a few clock cycles after reset
        #30;
        // reset = 0;
        // mac_enable = 1;
        // iact = 32'b10010000101100001000100001000001;
        iact = 32'hffffffff;
        msb_iact = 1;
        finish_iact_n = 1;
        #10;
        iact = 32'b0;
        msb_iact = 0;
        #10;
        iact = 32'hffffffff;
        #10;
        iact = 32'b0;
        #10;
        iact = 32'hffffffff;
        #10;
        iact = 32'b0;
        #10;
        iact = 32'hffffffff;
        #10;
        iact = 32'b0;
        #10;
        finish_iact_n = 0;
        #10;
        finish_iact_n = 1;
        #30;
        // msb_iact = 0;
        // #10;
        // reset = 1;
        // #10;
        // reset = 0;
        // mac_enable = 1;
        // iact = 1;
        // msb_iact = 1;
        // #10;
        // msb_iact = 0;
        // #10;
        $display("%t\t%h", $time, oact);
        $finish;
    end

  initial begin
    `ifdef VCS
        $fsdbDumpfile("dump.fsdb");
        $fsdbDumpvars(0, "+all");
    `endif
  end
endmodule

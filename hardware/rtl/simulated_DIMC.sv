/*
Chip-verified behavioral DIMC model for simulation and numerical verification.
*/

/* Computation:
iact^T * weight = oact
*/
import microarch_parameters::*;
module dimc_bank_row_acc_32x32_wReg # (
    localparam NUM_BASIC_WEIGHT_COL = DIMC_NUM_COL / DIMC_BASIC_WEIGHT_WIDTH,  // number of weight columns there, = number of shift_regs
    localparam WEIGHT_CONFIG_WIDTH = $clog2(DIMC_NUM_WEIGHT_CONFIG)
)
(
    input logic clk,
    // input reset,
    input logic [DIMC_ADDR_WIDTH-1:0] write_addr_in,
    input logic [DIMC_NUM_COL-1:0] write_data_in,
    input logic [DIMC_NUM_ROW-1:0] activation_in,  // input activation, this is a 1b vector
    input logic write_en_in,
    input logic [WEIGHT_CONFIG_WIDTH-1:0] weight_config_in, // 4b, or 8b
    input logic MSB_in, // most significant bit of input activation, used for sign extension
    input logic finish_n_in,
    output logic [DIMC_DATA_OUT_WIDTH-1:0] Output_out // output activation
);
    logic [DIMC_ADDR_WIDTH-1:0] addr_reg;
    logic [DIMC_NUM_COL-1:0] weight_in_reg;
    logic [DIMC_NUM_ROW-1:0] iact_reg;  // input activation, this is a 1b vector
    logic write_enable_reg;
    // input mac_enable,  // no need, replaced by MSB and LSB_n
    logic [WEIGHT_CONFIG_WIDTH-1:0] weight_config_reg; // 4b, or 8b
    logic msb_iact_reg; // most significant bit of logic activation, used for sign extension
    logic finish_iact_n_reg;
    logic [DIMC_DATA_OUT_WIDTH-1:0] oact_reg; // output activation

always_ff @ (posedge clk) begin
    addr_reg <= write_addr_in;
    weight_in_reg <= write_data_in;
    iact_reg <= activation_in;
    write_enable_reg <= write_en_in;
    weight_config_reg <= weight_config_in;
    msb_iact_reg <= MSB_in;
    finish_iact_n_reg <= finish_n_in;
    Output_out <= oact_reg;
end

logic [DIMC_NUM_COL-1:0] array [DIMC_NUM_ROW];
logic [16:0] shift_regs [NUM_BASIC_WEIGHT_COL];  // TODO: parameterize 16?

integer i;
integer j;
integer k;
integer flag_signed_weight;  // actually boolean type
integer temp_data_out;

logic [DIMC_MAX_OACT_WIDTH-1:0] weight_8b_output [NUM_BASIC_WEIGHT_COL/2];
logic [20:0] upper_half_results [4];
always_comb
    for (int n=0; n < 4; n++)
        upper_half_results[n] = ({shift_regs[2*n+1], 4'b0});

logic [20:0] lower_half_results [4];
always_comb
    for (int n=0; n < 4; n++)
         lower_half_results[n] = $signed(shift_regs[2*n]);

logic [20:0] full_results [4];
always_comb
    for (int n=0; n < 4; n++)
        full_results[n] = upper_half_results[n] + lower_half_results[n];

always_comb begin
    for (k = 0; k < NUM_BASIC_WEIGHT_COL / 2; k=k+1) begin
        // weight_8b_output[k] = $signed(({4'b0, shift_regs[2*k+1]} << 4) & 20'hFFFFF) + $signed(shift_regs[2*k]);  // get the last 20 bits
        weight_8b_output[k] = $signed(full_results[k]);
        // weight_8b_output[k] = $signed(($signed({shift_regs[2*k+1]}) << 4)) + $signed(shift_regs[2*k]);
    end
end

assign oact_reg[DIMC_MAX_OACT_WIDTH-1:0] = weight_config_reg == 1'b1 ? {shift_regs[1], shift_regs[0]}: weight_8b_output[0][DIMC_MAX_OACT_WIDTH-1:0]; 
assign oact_reg[2*DIMC_MAX_OACT_WIDTH-1:DIMC_MAX_OACT_WIDTH] = weight_config_reg == 1'b1 ? {shift_regs[3], shift_regs[2]}: weight_8b_output[1][DIMC_MAX_OACT_WIDTH-1:0];
assign oact_reg[3*DIMC_MAX_OACT_WIDTH-1:2*DIMC_MAX_OACT_WIDTH] = weight_config_reg == 1'b1 ? {shift_regs[5], shift_regs[4]}: weight_8b_output[2][DIMC_MAX_OACT_WIDTH-1:0];
assign oact_reg[4*DIMC_MAX_OACT_WIDTH-1:3*DIMC_MAX_OACT_WIDTH] = weight_config_reg == 1'b1 ? {shift_regs[7], shift_regs[6]}: weight_8b_output[3][DIMC_MAX_OACT_WIDTH-1:0];

always_ff @(posedge clk) begin
    if (write_enable_reg) begin
        for (j = 0; j < NUM_BASIC_WEIGHT_COL; j=j+1)
            shift_regs[j] <= 0;
        array[addr_reg] <= weight_in_reg;
        temp_data_out = 0;
    end
    else begin
        for (j = 0; j < NUM_BASIC_WEIGHT_COL; j=j+1) begin
            if (weight_config_reg == 0)
                flag_signed_weight = (j % 2 == 1);
            else begin
                flag_signed_weight = 1;  // always 1 for 4b
            end
            temp_data_out = 0; // Temporary variable for accumulation
            for (i = 0; i < DIMC_NUM_ROW; i=i+1) begin
                if (flag_signed_weight) begin
                    // VCS will treat X * 0 to be X, have to treat iact_reg[i] == 0 separately to avoid X propagation
                    temp_data_out = temp_data_out + ((iact_reg[i]==1'b0)? 0 : $signed($signed(array[i][DIMC_BASIC_WEIGHT_WIDTH*j +: DIMC_BASIC_WEIGHT_WIDTH]) * iact_reg[i]));
                    // $display("j%d, prod in signed = %0d, iact[%0d] = ", j, $signed($signed(array[i][DIMC_BASIC_WEIGHT_WIDTH*j +: DIMC_BASIC_WEIGHT_WIDTH]) * iact[i]), i, iact[i]);
                    // $display("j%d, temp_data_out in signed = %d", j, temp_data_out);
                end
                else begin
                    temp_data_out = temp_data_out + ((iact_reg[i]==1'b0)? 0 : (array[i][DIMC_BASIC_WEIGHT_WIDTH*j +: DIMC_BASIC_WEIGHT_WIDTH] * iact_reg[i]));
                    // $display("j%d, temp_data_out in unsigned = %d", j, temp_data_out);
                end
            end
            // $display("time = %0t, temp_data_out = %d, msb_iact = %b", $time, temp_data_out, msb_iact);
            if (msb_iact_reg) begin
            // if (msb_iact & flag_signed_weight) begin
                temp_data_out = ~ temp_data_out; // Negate
                // $display("time = %0t, temp_data_out after negate = %d", $time, temp_data_out);
            end
            // $display("temp_data_out = %d", temp_data_out);
            shift_regs[j] <= finish_iact_n_reg == 1? $signed(temp_data_out + (shift_regs[j] << 1) + msb_iact_reg) : 0;
            // $display("time = %0t, shift_regs value = %d", $time, $signed(temp_data_out + (shift_regs[j] << 1) + msb_iact));
            // $display("time = %0t, shift_regs[%d] = %d", $time, j, $signed(shift_regs[j]));
        end
    end

end
endmodule

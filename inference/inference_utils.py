def extract_level_representation(llm_output, model_type="llama-3", orientation="horizontal", separator="\n", empty_space='-'):
    if isinstance(llm_output, list):
        llm_output = llm_output[0]

    level_content = ""
    if model_type == "llama-3":
        parts = llm_output.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) > 1:
            assistant_section = parts[-1]

            if "<|eot_id|>" in assistant_section:
                level_content = assistant_section.split("<|eot_id|>")[0].strip()
            else:
                level_content = assistant_section.strip()
        else:
            level_content = llm_output.strip()

    elif model_type == "gemma-3":
        parts = llm_output.split("<start_of_turn>model")

        if len(parts) > 1:
            model_section = parts[-1]

            if "<end_of_turn>" in model_section:
                level_content = model_section.split("<end_of_turn>")[0].strip()
            else:
                level_content = model_section.strip()
        else:
            level_content = llm_output.strip()

    elif model_type == "qwen-2.5":
        parts = llm_output.split("<|im_start|>assistant")

        if len(parts) > 1:
            assistant_block = parts[-1]

            if "<|im_end|>" in assistant_block:
                level_content = assistant_block.split("<|im_end|>")[0].strip()
            else:
                level_content = assistant_block.strip()
        else:
            level_content = llm_output.strip()

    elif model_type == "qwen-3":
        parts = llm_output.split("<|im_start|>assistant")

        if len(parts) > 1:
            assistant_block = parts[-1]

            if "<|im_end|>" in assistant_block:
                content = assistant_block.split("<|im_end|>")[0].strip()

                if "<think>" in content and "</think>" in content:
                    think_parts = content.split("</think>")
                    if len(think_parts) > 1:
                        level_content = think_parts[-1].strip()
                    else:
                        level_content = content
                else:
                    level_content = content
            else:
                level_content = assistant_block.strip()
        else:
            level_content = llm_output.strip()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print("New level content:")
    print(level_content)

    if "|" in level_content and "\n" not in level_content:
        separator = "|"
    elif "\n" in level_content and "|" not in level_content:
        separator = "\n"

    if orientation == "vertical":
        level_content = VerticalLevel.reconstruct_level_from_vertical_bar(level_content, separator)
        
    return level_content


def fix_level_format(level_str, orientation="horizontal", separator="\n", empty_space='-'):
    if orientation == "vertical":
        level_str = VerticalLevel.reconstruct_level_from_vertical_bar(level_str, separator)
    
    lines = level_str.split('\n')

    line_lengths = [len(line) for line in lines]

    changed = True
    while changed:
        changed = False
        max_length = max(line_lengths)
        longest_lines_indices = [i for i, length in enumerate(line_lengths) if length == max_length]

        lines_trimmed = False
        for idx in longest_lines_indices:
            line = lines[idx]
            if line and line[-1] == empty_space:
                lines[idx] = line[:-1]
                line_lengths[idx] -= 1
                lines_trimmed = True
                changed = True

        if not lines_trimmed:
            break

    max_length = max(line_lengths)

    for i in range(len(lines)):
        if line_lengths[i] < max_length:
            padding_char = empty_space

            lines[i] = lines[i] + (padding_char * (max_length - line_lengths[i]))

    return '\n'.join(lines)

##############################################################################################

def fix_level_format_extra(level_str, orientation="horizontal", separator="\n", empty_space='-', 
                     line_quantity=None, column_quantity=None, enforce_shape=None, add_ground=None,
                     use_original_logic_on_column=False):

    if "|" in level_str and "\n" not in level_str:
        separator = "|"
    elif "\n" in level_str and "|" not in level_str:
        separator = "\n"

    if orientation == "vertical":
        level_str = VerticalLevel.reconstruct_level_from_vertical_bar(level_str, separator)
    
    lines = level_str.split(separator)
    
    # Handle line quantity enforcement
    if enforce_shape in ["line", "both"] and line_quantity is not None:
        if len(lines) > line_quantity:
            # Remove from top (beginning of list)
            lines = lines[-line_quantity:]
        elif len(lines) < line_quantity:
            # Add empty lines at top
            empty_line = empty_space * (max(len(line) for line in lines) if lines else column_quantity or 0)
            lines = [empty_line] * (line_quantity - len(lines)) + lines
    
    # Handle column quantity enforcement
    column_adjusted = False
    if enforce_shape in ["column", "both"] and column_quantity is not None:
        column_adjusted = True
        for i in range(len(lines)):
            if len(lines[i]) > column_quantity:
                lines[i] = lines[i][:column_quantity]
            elif len(lines[i]) < column_quantity:
                if i == len(lines) - 1 and add_ground is not None:
                    lines[i] = lines[i] + (add_ground * (column_quantity - len(lines[i])))
                else:
                    lines[i] = lines[i] + (empty_space * (column_quantity - len(lines[i])))
    
    if (enforce_shape is None or use_original_logic_on_column) and not column_adjusted:
        line_lengths = [len(line) for line in lines]
        
        changed = True
        while changed:
            changed = False
            max_length = max(line_lengths)
            longest_lines_indices = [i for i, length in enumerate(line_lengths) if length == max_length]
            
            lines_trimmed = False
            for idx in longest_lines_indices:
                line = lines[idx]
                if line and line[-1] == empty_space:
                    lines[idx] = line[:-1]
                    line_lengths[idx] -= 1
                    lines_trimmed = True
                    changed = True
            
            if not lines_trimmed:
                break
        
        max_length = max(line_lengths)
        for i in range(len(lines)):
            if line_lengths[i] < max_length:
                padding_char = add_ground if i == len(lines) - 1 and add_ground is not None else empty_space
                lines[i] = lines[i] + (padding_char * (max_length - line_lengths[i]))
    
    return separator.join(lines)

##############################################################################################

class VerticalLevel:
    @staticmethod
    def reconstruct_level_from_vertical_bar(vertical_bar_str, separator="\n"):
        if not vertical_bar_str:
            return None

        vertical_columns = vertical_bar_str.split(separator)
        num_cols = len(vertical_columns)

        if num_cols == 0 or not vertical_columns[0]:
            return None

        num_rows = len(vertical_columns[0])
        if num_rows == 0:
            return None 

        reconstructed_rows = []
        for i in range(num_rows): 
            current_row_chars = []
            for j in range(num_cols): 
                vertical_char_index = num_rows - 1 - i
                if j < len(vertical_columns) and vertical_char_index < len(vertical_columns[j]):
                    char = vertical_columns[j][vertical_char_index]
                    current_row_chars.append(char)

            reconstructed_rows.append("".join(current_row_chars))

        return "\n".join(reconstructed_rows)
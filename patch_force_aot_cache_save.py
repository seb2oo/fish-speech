# from pathlib import Path
# import ast

# TARGET = Path(
#     "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py"
# )

# BACKUP = TARGET.with_suffix(".py.backup_force_aot_cache_save")

# OLD = """    cache_info = aot_config.cache_info
#     if cache_info is not None:
#         if hasattr(compiled_fw, "_fx_graph_cache_key"):
#             time_taken_ns = time.time_ns() - cache_info.start_time_ns
#             guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
#             entry = AOTAutogradCache.make_entry(
#                 compiled_fw_func=compiled_fw,  # type: ignore[arg-type]
#                 compiled_bw_func=None,
#                 aot_joint_graph_str=None,
#                 aot_forward_graph_str=aot_forward_graph_str,
#                 aot_backward_graph_str=None,
#                 runtime_metadata=fw_metadata,
#                 dispatch_wrappers=wrappers,
#                 maybe_subclass_meta=maybe_subclass_meta,
#                 num_fw_outs_saved_for_bw=None,
#                 indices_of_inps_to_detach=[],
#                 forward_time_taken_ns=time_taken_ns,
#                 backward_time_taken_ns=0,
#                 sanitized_aot_config=sanitize_aot_config(aot_config),
#                 guards_expr=guards_expr,
#                 backward_state_indices=None,
#                 num_symints_saved_for_bw=None,
#                 serialized_bw_module=None,
#             )
#             AOTAutogradCache.save(
#                 cache_info.cache_key, entry, remote=should_use_remote_autograd_cache()
#             )
# """

# NEW = """    cache_info = aot_config.cache_info
#     if cache_info is not None:
#         time_taken_ns = time.time_ns() - cache_info.start_time_ns
#         guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
#         entry = AOTAutogradCache.make_entry(
#             compiled_fw_func=compiled_fw,  # type: ignore[arg-type]
#             compiled_bw_func=None,
#             aot_joint_graph_str=None,
#             aot_forward_graph_str=aot_forward_graph_str,
#             aot_backward_graph_str=None,
#             runtime_metadata=fw_metadata,
#             dispatch_wrappers=wrappers,
#             maybe_subclass_meta=maybe_subclass_meta,
#             num_fw_outs_saved_for_bw=None,
#             indices_of_inps_to_detach=[],
#             forward_time_taken_ns=time_taken_ns,
#             backward_time_taken_ns=0,
#             sanitized_aot_config=sanitize_aot_config(aot_config),
#             guards_expr=guards_expr,
#             backward_state_indices=None,
#             num_symints_saved_for_bw=None,
#             serialized_bw_module=None,
#         )
#         AOTAutogradCache.save(
#             cache_info.cache_key, entry, remote=should_use_remote_autograd_cache()
#         )
# """

# if not TARGET.exists():
#     raise FileNotFoundError(TARGET)

# source = TARGET.read_text()

# if OLD not in source:
#     raise RuntimeError(
#         "Expected original AOT cache block not found. "
#         "Nothing was modified."
#     )

# if not BACKUP.exists():
#     BACKUP.write_text(source)
#     print("[OK] Backup created:", BACKUP)
# else:
#     print("[OK] Backup already exists:", BACKUP)

# patched = source.replace(OLD, NEW, 1)

# ast.parse(patched)

# TARGET.write_text(patched)

# print("[OK] Removed _fx_graph_cache_key gate")
# print("[OK] Corrected indentation of complete cache-save block")
# print("[OK] Syntax check passed")
# print("[OK] Patch written successfully")








# from pathlib import Path
# import ast

# TARGET = Path(
#     "/usr/local/lib/python3.12/dist-packages/torch/_functorch/"
#     "_aot_autograd/jit_compile_runtime_wrappers.py"
# )

# BACKUP = TARGET.with_suffix(
#     ".py.backup_force_aot_cache_save"
# )

# OLD = """    cache_info = aot_config.cache_info
#     if cache_info is not None:
#         if hasattr(compiled_fw, "_fx_graph_cache_key"):
#             time_taken_ns = time.time_ns() - cache_info.start_time_ns
#             guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
#             entry = AOTAutogradCache.make_entry(
#                 compiled_fw_func=compiled_fw,
#                 compiled_bw_func=None,
#                 aot_joint_graph_str=None,
#                 aot_forward_graph_str=aot_forward_graph_str,
#                 aot_backward_graph_str=None,
#                 runtime_metadata=fw_metadata,
#                 dispatch_wrappers=wrappers,
#                 maybe_subclass_meta=maybe_subclass_meta,
#                 num_fw_outs_saved_for_bw=None,
#                 indices_of_inps_to_detach=[],
#                 forward_time_taken_ns=time_taken_ns,
#                 backward_time_taken_ns=0,
#                 sanitized_aot_config=sanitize_aot_config(aot_config),
#                 guards_expr=guards_expr,
#                 backward_state_indices=None,
#                 num_symints_saved_for_bw=None,
#                 serialized_bw_module=None,
#             )
#             AOTAutogradCache.save(
#                 cache_info.cache_key,
#                 entry,
#                 remote=should_use_remote_autograd_cache(),
#             )
# """

# NEW = """    cache_info = aot_config.cache_info
#     if cache_info is not None:
#         time_taken_ns = time.time_ns() - cache_info.start_time_ns
#         guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
#         entry = AOTAutogradCache.make_entry(
#             compiled_fw_func=compiled_fw,
#             compiled_bw_func=None,
#             aot_joint_graph_str=None,
#             aot_forward_graph_str=aot_forward_graph_str,
#             aot_backward_graph_str=None,
#             runtime_metadata=fw_metadata,
#             dispatch_wrappers=wrappers,
#             maybe_subclass_meta=maybe_subclass_meta,
#             num_fw_outs_saved_for_bw=None,
#             indices_of_inps_to_detach=[],
#             forward_time_taken_ns=time_taken_ns,
#             backward_time_taken_ns=0,
#             sanitized_aot_config=sanitize_aot_config(aot_config),
#             guards_expr=guards_expr,
#             backward_state_indices=None,
#             num_symints_saved_for_bw=None,
#             serialized_bw_module=None,
#         )
#         AOTAutogradCache.save(
#             cache_info.cache_key,
#             entry,
#             remote=should_use_remote_autograd_cache(),
#         )
# """

# if not TARGET.exists():
#     raise FileNotFoundError(TARGET)

# source = TARGET.read_text()

# if NEW in source:
#     print("[INFO] Patch already installed")
#     raise SystemExit(0)

# if OLD not in source:
#     raise RuntimeError(
#         "Expected original AOT cache block not found. "
#         "Nothing was modified."
#     )

# if not BACKUP.exists():
#     BACKUP.write_text(source)
#     print("[OK] Backup created:", BACKUP)
# else:
#     print("[OK] Backup already exists:", BACKUP)

# patched = source.replace(OLD, NEW, 1)

# ast.parse(patched)

# TARGET.write_text(patched)

# print("[OK] Removed _fx_graph_cache_key gate")
# print("[OK] AOT cache-save block corrected")
# print("[OK] Syntax check passed")
# print("[OK] Patch written successfully")




from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/"
    "_aot_autograd/jit_compile_runtime_wrappers.py"
)

BACKUP = TARGET.with_suffix(
    ".py.backup_force_aot_cache_save"
)

OLD = """    cache_info = aot_config.cache_info
    if cache_info is not None:
        if hasattr(compiled_fw, "_fx_graph_cache_key"):
            time_taken_ns = time.time_ns() - cache_info.start_time_ns
            guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
            entry = AOTAutogradCache.make_entry(
                compiled_fw_func=compiled_fw,  # type: ignore[arg-type]
                compiled_bw_func=None,
                aot_joint_graph_str=None,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=None,
                runtime_metadata=fw_metadata,
                dispatch_wrappers=wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=None,
                indices_of_inps_to_detach=[],
                forward_time_taken_ns=time_taken_ns,
                backward_time_taken_ns=0,
                sanitized_aot_config=sanitize_aot_config(aot_config),
                guards_expr=guards_expr,
                backward_state_indices=None,
                num_symints_saved_for_bw=None,
                serialized_bw_module=None,
            )
            AOTAutogradCache.save(
                cache_info.cache_key, entry, remote=should_use_remote_autograd_cache()
            )
"""

NEW = """    cache_info = aot_config.cache_info
    if cache_info is not None:
        time_taken_ns = time.time_ns() - cache_info.start_time_ns
        guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
        entry = AOTAutogradCache.make_entry(
            compiled_fw_func=compiled_fw,  # type: ignore[arg-type]
            compiled_bw_func=None,
            aot_joint_graph_str=None,
            aot_forward_graph_str=aot_forward_graph_str,
            aot_backward_graph_str=None,
            runtime_metadata=fw_metadata,
            dispatch_wrappers=wrappers,
            maybe_subclass_meta=maybe_subclass_meta,
            num_fw_outs_saved_for_bw=None,
            indices_of_inps_to_detach=[],
            forward_time_taken_ns=time_taken_ns,
            backward_time_taken_ns=0,
            sanitized_aot_config=sanitize_aot_config(aot_config),
            guards_expr=guards_expr,
            backward_state_indices=None,
            num_symints_saved_for_bw=None,
            serialized_bw_module=None,
        )
        AOTAutogradCache.save(
            cache_info.cache_key, entry, remote=should_use_remote_autograd_cache()
        )
"""

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

source = TARGET.read_text()

if NEW in source:
    print("[INFO] Patch already installed")
    raise SystemExit(0)

if OLD not in source:
    raise RuntimeError(
        "Expected original AOT cache block not found. "
        "Nothing was modified."
    )

if not BACKUP.exists():
    BACKUP.write_text(source)
    print("[OK] Backup created:", BACKUP)
else:
    print("[OK] Backup already exists:", BACKUP)

patched = source.replace(OLD, NEW, 1)

ast.parse(patched)

TARGET.write_text(patched)

print("[OK] Removed _fx_graph_cache_key gate")
print("[OK] AOT cache-save block corrected")
print("[OK] Syntax check passed")
print("[OK] Patch written successfully")
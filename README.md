# lag-llama

## 最近更新

### 2024-11-23: hours_stock_status 字段处理优化
- 修改了数据处理逻辑，将 `hours_stock_status` 字段与 `hours_sale` 字段采用相同的处理方式
- 所有 `list<numeric>` 和 `list<binary>` 类型的字段现在都会被展开成独立的特征列
- `hours_stock_status` 的24小时数据现在被拆分成24个独立的特征列（`hours_stock_status__00` 到 `hours_stock_status__23`）
- `hours_sale` 的24小时数据也被拆分成24个独立的特征列（`hours_sale__00` 到 `hours_sale__23`）

## Git 命令参考
```bash
git log --oneline -- Demo1.py    # 查看目标文件更改记录
git diff i7j8k9l a1b2c3d main.py    # 对比两次修改情况
```

## 启动数据处理
```powershell
python scripts/prepare_custom_dataset.py `
  --schema datasets/字段_v3.csv `
  --data datasets/门店商品数据 `
  --output datasets/store_product_json `
  --freq D `
  --prediction-length 28 `
  --group-keys city_id store_id product_id `
  --date-col dt `
  --target-col sale_amount
```
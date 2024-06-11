import yaml
import create_datasets  # 确保用实际模块名替换 your_module
import settin_and_training

def main():
    # 读取配置文件
    with open('create_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    dataset_creator = create_datasets.Create_datasets(
        excel_name=config['excel_name'],
        csv_name=config['csv_name'],
        datasets_name=config['datasets_name'],
        lang=config['lang'],
    )

    dataset_creator.creat_excel()
    df = dataset_creator.creat_datasets(total=config['total'])
    dataset_creator.continue_translate_xlsx(df)
    print('==========================================================================================================')
    print("create_datasets are completed.")
    print('==========================================================================================================')
    
    model_use = settin_and_training.ViTLSTM()
    train_loader, dataset = model_use.get_datasets(config['csv_name'])
    print('==========================================================================================================')
    print('start training.')
    print('==========================================================================================================')
    model = model_use.setting_and_training(train_loader, dataset)

    print('==========================================================================================================')
    print('training endding.')
    print('==========================================================================================================')
    model_use.save(model)


    print('==========================================================================================================')
    print('All task are completed.')
    print('==========================================================================================================')


if __name__ == "__main__":
    main()

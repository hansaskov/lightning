from lightning.pytorch import Trainer
from model import NN
from dataset import kickBallDataModule
from lightning.pytorch.loggers import CSVLogger
import config


if __name__ == "__main__":
    '''
    utilities.calc_mean_std(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        num_channels=config.NUM_CHANNELS   
    )
    '''
 
    model = NN(
        input_size=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
        class_weighting=config.CLASS_WEIGHTING
    )
    dm = kickBallDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_height=config.IMAGE_HEIGHT,
        img_width=config.IMAGE_WIDTH,
        data_mean=config.DATA_MEAN,
        data_std=config.DATA_STD,
    )
    trainer = Trainer(
        accelerator=config.ACCELERATOR,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
        logger= CSVLogger(save_dir=config.LOGS_FOLDER)
    )   

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
import torch
import numpy as np


def fit(train_loader, 
        val_loader, 
        model, 
        loss_fn, 
        softmax_loss_fn, 
        optimizer, 
        scheduler, 
        n_epochs, 
        cuda,
        log_interval,
        metrics=[],
        start_epoch=0,
        multi_class=False,
        softmax=False):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    for epoch in range(start_epoch, n_epochs):
        
        # Train stage
        if multi_class:
            train_loss, metrics = multi_class_train_epoch(train_loader,
                                                          model,
                                                          loss_fn,
                                                          optimizer,
                                                          cuda,
                                                          log_interval,
                                                          metrics)
        else:    
            if softmax:
                train_loss, metrics = softmax_triplet_train_epoch(train_loader,
                                                                  model,
                                                                  loss_fn,
                                                                  softmax_loss_fn,
                                                                  optimizer,
                                                                  cuda,
                                                                  log_interval,
                                                                  metrics)
            else:    
                train_loss, metrics = train_epoch(train_loader,
                                                  model,
                                                  loss_fn,
                                                  optimizer,
                                                  cuda,
                                                  log_interval,
                                                  metrics)

        scheduler.step()
    
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        if multi_class:
            val_loss, metrics = multi_class_test_epoch(val_loader, model, loss_fn, cuda, metrics)
            val_loss /= len(val_loader)
        else:
            val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
            val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
             message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader,
                model,
                loss_fn,
                optimizer,
                cuda,
                log_interval,
                metrics):
    
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    #for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx, (a, b, data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def softmax_triplet_train_epoch(train_loader,
                                model,
                                loss_fn,
                                softmax_loss_fn,
                                optimizer,
                                cuda,
                                log_interval,
                                metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    for batch_idx, (face, flank, full, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(full) in (tuple, list):
            full = (full,)
        if cuda:
            full = tuple(d.cuda() for d in full)
            if target is not None:
                target = target.cuda()

        # Set the gradient to 0
        optimizer.zero_grad()
        
        #Run the model to generate triplet embeddings and softmax out
        triplet_outputs, softmax_outputs = model(*full)

        #Compute cross entropy loss with softmax
        softmax_loss = softmax_loss_fn(softmax_outputs, target) 
        
        #Backprop with softmax
        softmax_loss.backward(retain_graph=True)

        #Compute triplet loss
        if type(triplet_outputs) not in (tuple, list):
            triplet_outputs = (triplet_outputs,)
      
        triplet_loss_inputs = triplet_outputs
        if target is not None:
            target = (target,)
            triplet_loss_inputs += target

        #compute triplet loss    
        triplet_loss_outputs = loss_fn(*triplet_loss_inputs)
        triplet_loss = triplet_loss_outputs[0] if type(triplet_loss_outputs) in (tuple, list) else triplet_loss_outputs
        
        #backprop triplet loss
        triplet_loss.backward(retain_graph=True)
        
        #Run optimizer
        optimizer.step()
        
        loss = triplet_loss + softmax_loss*1.0/softmax_outputs.size()[0]
        losses.append(loss.item())
        total_loss += loss.item()
     
        for metric in metrics:
            metric(triplet_outputs, target, triplet_loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(full[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def multi_class_train_epoch(train_loader,
                            model,
                            loss_fn,
                            optimizer,
                            cuda,
                            log_interval,
                            metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data_face, data_flank, data_full, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data_face) in (tuple, list):
            data_face = (data_face,)
        if not type(data_flank) in (tuple, list):
            data_flank = (data_flank,)
        if not type(data_full) in (tuple, list):
            data_full = (data_full,)    
            
        if cuda:
            data_face = tuple(d.cuda() for d in data_face)
            data_flank = tuple(d.cuda() for d in data_flank)
            data_full = tuple(d.cuda() for d in data_full)    
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data_face, *data_flank, *data_full)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data_face[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader,
               model,
               loss_fn,
               cuda,
               metrics):
    
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        #for batch_idx, (data, target) in enumerate(val_loader):
        for batch_idx, (a, b, data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


def multi_class_test_epoch(val_loader,
                            model,
                            loss_fn,
                            cuda,
                            metrics):
    for metric in metrics:
        metric.reset()

    val_loss = 0    
    model.eval()
       

    with torch.no_grad():    
        
        for batch_idx, (data_face, data_flank, data_full, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data_face) in (tuple, list):
                data_face = (data_face,)
            if not type(data_flank) in (tuple, list):
                data_flank = (data_flank,)
            if not type(data_full) in (tuple, list):
                data_full = (data_full,)    
            
            if cuda:
                data_face = tuple(d.cuda() for d in data_face)
                data_flank = tuple(d.cuda() for d in data_flank)
                data_full = tuple(d.cuda() for d in data_full)    
                if target is not None:
                    target = target.cuda()

            outputs = model(*data_face, *data_flank, *data_full)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics

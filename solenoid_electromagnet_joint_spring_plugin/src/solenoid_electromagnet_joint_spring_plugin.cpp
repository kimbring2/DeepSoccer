#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
	class TorsionalSpringPlugin : public ModelPlugin
	{
		public: TorsionalSpringPlugin() {}

		private:
			physics::ModelPtr model;
			sdf::ElementPtr sdf;
			
			physics::JointPtr joint;

			// Set point
			double setPoint;

			// Spring constant
			double kx;

			// Pointer to update event connection
			event::ConnectionPtr updateConnection;

		public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
		{
			// Safety check
			if (_model->GetJointCount() == 0)
			{
				std::cerr << "You have zero joints! Something is wrong! Not loading plugin." << std::endl;
				return;
			}

			// Store model pointer
			this->model = _model;

			// Store the SDF pointer
			this->sdf = _sdf;

			if (_sdf->HasElement("joint"))
				this->joint = _model->GetJoint(_sdf->Get<std::string>("joint"));
			else
				std::cerr << "Must specify joint to apply a torsional spring at!\n";

			this->kx = 0.0;
			if (_sdf->HasElement("kx"))
				this->kx = _sdf->Get<double>("kx");
			else
				printf("Torsional spring coefficient not specified! Defaulting to: %f\n", this->kx);

			this->setPoint = 0.0;

			if (_sdf->HasElement("set_point"))
				this->setPoint = _sdf->Get<double>("set_point");
			else
				printf("Set point not specified! Defaulting to: %f\n", this->setPoint);

			std::cout << "Loaded gazebo_joint_torsional_spring." << std::endl;

		}

		public: void Init()
		{
			// Listen to update event
			this->updateConnection = event::Events::ConnectWorldUpdateBegin(
				std::bind (&TorsionalSpringPlugin::OnUpdate, this) );
		}

		protected: void OnUpdate()
		{
			double current_angle = this->joint->Position(0);
			std::cout << "current_angle: " << current_angle << std::endl;
			this->joint->SetForce(0, (this->kx * (this->setPoint - current_angle)) );			
		}
		
	};

	GZ_REGISTER_MODEL_PLUGIN(TorsionalSpringPlugin)
}

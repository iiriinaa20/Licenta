from dataclasses import asdict, dataclass, field
from typing import List
from datetime import datetime

@dataclass
class User:
    email: str
    name: str
    type: str  # either 'teacher' or 'student'

@dataclass
class Course:
    name: str
    credits: int
    year: int
    semester: str  # e.g., 'Fall', 'Spring'

@dataclass
class CoursesPlanification:
    user_id: str
    course_id: str
    date: datetime
    start_time: datetime
    end_time: datetime

@dataclass
class Attendance:
    user_id: str
    course_id: str
    attendance: List[datetime] = field(default_factory=list)

class UserRepository:
    db_con = None
    collection_name = 'users'

    @staticmethod
    def create(data):
        user = User(**data)
        UserRepository.db_con.db.collection(UserRepository.collection_name).add(asdict(user))

    @staticmethod
    def read(user_id):
        user_ref = UserRepository.db_con.db.collection(UserRepository.collection_name).document(user_id)
        user = user_ref.get()
        return user.to_dict() if user.exists else None

    @staticmethod
    def update(user_id, data):
        user_ref = UserRepository.db_con.db.collection(UserRepository.collection_name).document(user_id)
        user_ref.update(data)

    @staticmethod
    def delete(user_id):
        user_ref = UserRepository.db_con.db.collection(UserRepository.collection_name).document(user_id)
        user_ref.delete()

class CourseRepository:
    db_con = None
    collection_name = 'courses'

    @staticmethod
    def create(data):
        course = Course(**data)
        CourseRepository.db_con.db.collection(CourseRepository.collection_name).add(asdict(course))

    @staticmethod
    def read(course_id):
        course_ref = CourseRepository.db_con.db.collection(CourseRepository.collection_name).document(course_id)
        course = course_ref.get()
        return course.to_dict() if course.exists else None
    
    def read_all():
        courses_ref = CourseRepository.db_con.db.collection(CourseRepository.collection_name).stream()
        return [{'id': course.id, **course.to_dict()} for course in courses_ref]

    @staticmethod
    def update(course_id, data):
        course_ref = CourseRepository.db_con.db.collection(CourseRepository.collection_name).document(course_id)
        course_ref.update(data)

    @staticmethod
    def delete(course_id):
        course_ref = CourseRepository.db_con.db.collection(CourseRepository.collection_name).document(course_id)
        course_ref.delete()

class CoursesPlanificationRepository:
    db_con = None
    collection_name = 'courses_planification'

    @staticmethod
    def create(data):
        courses_planification = CoursesPlanification(**data)
        CoursesPlanificationRepository.db_con.db.collection(CoursesPlanificationRepository.collection_name).add(asdict(courses_planification))

    @staticmethod
    def read(plan_id):
        plan_ref = CoursesPlanificationRepository.db_con.db.collection(CoursesPlanificationRepository.collection_name).document(plan_id)
        plan = plan_ref.get()
        return plan.to_dict() if plan.exists else None

    @staticmethod
    def update(plan_id, data):
        plan_ref = CoursesPlanificationRepository.db_con.db.collection(CoursesPlanificationRepository.collection_name).document(plan_id)
        plan_ref.update(data)

    @staticmethod
    def delete(plan_id):
        plan_ref = CoursesPlanificationRepository.db_con.db.collection(CoursesPlanificationRepository.collection_name).document(plan_id)
        plan_ref.delete()

class AttendanceRepository:
    db_con = None
    collection_name = 'attendance'

    @staticmethod
    def create(data):
        attendance = Attendance(**data)
        AttendanceRepository.db_con.db.collection(AttendanceRepository.collection_name).add(asdict(attendance))

    @staticmethod
    def read(attendance_id):
        attendance_ref = AttendanceRepository.db_con.db.collection(AttendanceRepository.collection_name).document(attendance_id)
        attendance = attendance_ref.get()
        return attendance.to_dict() if attendance.exists else None

    @staticmethod
    def update(attendance_id, data):
        attendance_ref = AttendanceRepository.db_con.db.collection(AttendanceRepository.collection_name).document(attendance_id)
        attendance_ref.update(data)

    @staticmethod
    def delete(attendance_id):
        attendance_ref = AttendanceRepository.db_con.db.collection(AttendanceRepository.collection_name).document(attendance_id)
        attendance_ref.delete()
